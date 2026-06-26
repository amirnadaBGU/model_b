import yaml
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import os
import glob
import numpy as np
import cv2
import re

# ___What to do___:
ANALYZE_VAL = False
ANALYZE_PREDICT = True
ANALYSE_2_STEP = False

# ___Global variables___:
IOU_THRESHOLD = 0.5 # what counts as match
CONF = 0.25 # model confidence score
GRAPHICAL_DEBUG = False # graphical debug
MODE = 'CLASSIC' #'ADVANCED' # maybe not relevant


# When True: PREDICT undergoes the same preprocessing as model.val() — rectangular letterbox to 672
# (640 native content + gray border, no upscale). When False: standard predict at 640.
USE_VAL_PREPROCESS = True
VAL_IMGSZ = 672

# class-agnostic NMS post-detection in PREDICT: removes overlapping boxes across all classes
# and keeps only the one with the highest confidence for each object. Crucial for counting
# (to avoid double-counting the same fish), as the model is end-to-end and does not perform NMS on its own.
# When False: the output remains AS IS.
USE_NMS = False
NMS_IOU = 0.5  # The IoU threshold above which a box is considered a duplicate and removed

# Two-step pipeline: YOLO -> NMS(per flag) -> crop each detection from the ORIGINAL image
# -> ConvNeXt classifier (best_ckpt.ckpt) reassigns class {background->drop, fish, partial}
# -> evaluate. Lets stage-2 fix YOLO's class confusion and reject false positives.

# Classifier params:
CKPT_PATH = "best_ckpt.ckpt"                              # ConvNeXt (Lightning ckpt; loaded without pl)
ORIG_IMAGES_DIR = "datasets/data_original/valid/images"  # full-res originals, for the crops
CROP_PADDING = 0.10                                       # expand each box by 10% before cropping

# Bypass OpenMP multiple initialization error (prevents crash due to duplicate libraries)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# ======================================================================
# שלב 2 — מסווג ConvNeXt (best_ckpt.ckpt). נטען בלי pytorch_lightning:
# בונים convnext_tiny, מחליפים ראש ל-Sequential(Dropout, Linear(.,3)),
# וטוענים את ה-state_dict (מסירים את הקידומת 'model.' של ה-LightningModule).
# מחלקות המסווג (לפי ImageFolder, אלפביתי): 0=background, 1=fish, 2=partial_fish.
# ======================================================================

# Classifier
def load_convnext_classifier(ckpt_path, device):
    import torch
    import torchvision.models as tvm
    model = tvm.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(in_features, 3),
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    # שומרים רק משקולות הרשת (model.*) — מסירים מצבי metrics וכו'
    weights = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"[2-STEP] load_state_dict — missing={list(missing)} unexpected={list(unexpected)}")
    return model.eval().to(device)

def _original_stem(stem):
    """מסיר סיומת Roboflow: 'name_jpg.rf.<hash>' -> 'name'."""
    return re.sub(r'[_.](?:jpg|jpeg|png)\.rf\.[a-f0-9]+$', '', stem, flags=re.IGNORECASE)

def build_orig_lookup(orig_dir):
    """ממפה stem בסיסי (ללא hash) -> נתיב קובץ, עבור תיקיית התמונות המקוריות."""
    lookup = {}
    for f in glob.glob(os.path.join(orig_dir, "*")):
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"):
            lookup[_original_stem(os.path.splitext(os.path.basename(f))[0])] = f
    return lookup

def expand_box(x1, y1, x2, y2, img_w, img_h, padding):
    """מרחיב תיבה ב-padding (אחוז מהממדים שלה), חתוך לגבולות התמונה."""
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * padding)), max(0, int(y1 - bh * padding)),
            min(img_w, int(x2 + bw * padding)), min(img_h, int(y2 + bh * padding)))

def classify_crops(model, transform, crops, device):
    """crops: רשימת תמונות BGR (np). מחזיר רשימת (cn_class, confidence).
    כל crop עובר resize ל-224x224 ואז הטרנספורם של ConvNeXt (כמו באימון)."""
    import torch
    from PIL import Image
    if not crops:
        return []
    batch = torch.stack([
        transform(Image.fromarray(cv2.cvtColor(
            cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)))
        for c in crops
    ])
    with torch.no_grad():
        probs = torch.softmax(model(batch.to(device)), dim=1)
        conf, pred = probs.max(dim=1)
    return list(zip(pred.cpu().tolist(), conf.cpu().tolist()))

# Evaluation functions:

def evaluate_class_specific(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=IOU_THRESHOLD):
    """
    שכפול נאמן של לוגיקת השיוך של Ultralytics (validator.match_predictions) — המנוע
    שמאחורי ה-Precision/Recall/F1 שמודפסים ב-model.val().

    ההבדל מ-evaluate_class_specific_confidence:
      • השיוך נעשה לפי IoU (מהגבוה לנמוך) ולא לפי Confidence.
      • מסתכלים על כל זוגות (GT, חיזוי) ביחד באופן גלובלי, לא חיזוי-אחרי-חיזוי.
    זהו שיוך מודע-מחלקה: זוגות שבהם מחלקת החיזוי ≠ מחלקת ה-GT מאופסים מראש, כך
    שחיזוי יכול להתאים רק ל-GT מאותה מחלקה.
    מחזירה image_metrics (TP/FP/FN לכל מחלקה) ו-pred_statuses ('tp'/'fp' לכל חיזוי).
    """
    image_metrics = {
        0: {"tp": 0, "fp": 0, "fn": 0},
        1: {"tp": 0, "fp": 0, "fn": 0}
    }

    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    pred_statuses = ['fp'] * n_pred

    # מקרי קצה
    if n_pred == 0:
        for g_cls in gt_classes:
            image_metrics[int(g_cls)]["fn"] += 1
        return image_metrics, pred_statuses

    if n_gt == 0:
        for p_cls in pred_classes:
            image_metrics[int(p_cls)]["fp"] += 1
        return image_metrics, pred_statuses

    # 1. מטריצת IoU בין כל GT (שורות) לכל חיזוי (עמודות) — המקבילה ל-box_iou ב-Ultralytics
    iou = np.zeros((n_gt, n_pred))
    for g in range(n_gt):
        for p in range(n_pred):
            iou[g, p] = calculate_iou(pred_boxes[p], gt_boxes[g])

    # 2. איפוס זוגות ממחלקות שונות => שיוך מודע-מחלקה (iou = iou * correct_class)
    correct_class = (np.asarray(gt_classes).astype(int)[:, None] ==
                     np.asarray(pred_classes).astype(int)[None, :])
    iou = iou * correct_class

    matched_gt = set()
    matched_pred = set()

    # 3. כל הזוגות שעוברים את סף ה-IoU, ממוינים מהגבוה לנמוך, ואז ייחוד לפי חיזוי ואז לפי GT
    #    (בדיוק כמו match_predictions: argsort על ה-IoU, ואז np.unique על עמודת החיזוי ואז על עמודת ה-GT)
    cand = np.argwhere(iou >= iou_threshold)  # שורות בפורמט [gt_idx, pred_idx]
    if len(cand) > 0:
        order = iou[cand[:, 0], cand[:, 1]].argsort()[::-1]
        cand = cand[order]
        cand = cand[np.unique(cand[:, 1], return_index=True)[1]]  # חיזוי יחיד לכל חיזוי
        cand = cand[np.unique(cand[:, 0], return_index=True)[1]]  # GT יחיד לכל GT

        for g_idx, p_idx in cand:
            g_idx, p_idx = int(g_idx), int(p_idx)
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            cls = int(pred_classes[p_idx])  # == מחלקת ה-GT, כי השיוך מודע-מחלקה
            image_metrics[cls]["tp"] += 1
            pred_statuses[p_idx] = 'tp'

    # 4. חיזויים שלא שויכו => FP (לפי מחלקת החיזוי)
    for p in range(n_pred):
        if p not in matched_pred:
            image_metrics[int(pred_classes[p])]["fp"] += 1
            pred_statuses[p] = 'fp'

    # 5. GT שלא שויכו => FN (לפי מחלקת ה-GT)
    for g in range(n_gt):
        if g not in matched_gt:
            image_metrics[int(gt_classes[g])]["fn"] += 1

    return image_metrics, pred_statuses

def evaluate_class_specific_conf(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=IOU_THRESHOLD):
    """
    מחשבת TP, FP, FN לכל מחלקה בנפרד ומחזירה גם מילון סטטוסים לציור גראפי.
    שיוך לפי סדר Confidence: החיזויים מגיעים ממוינים מהגבוה לנמוך, וכל חיזוי בתורו
    תופס את ה-GT הפנוי מאותה מחלקה בעל ה-IoU הכי גבוה.
    """
    image_metrics = {
        0: {"tp": 0, "fp": 0, "fn": 0},
        1: {"tp": 0, "fp": 0, "fn": 0}
    }

    # רשימה בגודל של pred_boxes שתחזיק 'tp' או 'fp' עבור כל ניבוי
    pred_statuses = ['fp'] * len(pred_boxes)

    if len(pred_boxes) == 0:
        for g_cls in gt_classes:
            image_metrics[int(g_cls)]["fn"] += 1
        return image_metrics, pred_statuses

    if len(gt_boxes) == 0:
        for p_cls in pred_classes:
            image_metrics[int(p_cls)]["fp"] += 1
        return image_metrics, pred_statuses

    for target_cls in [0, 1]:
        # indices of racing bounding boxes for the current class p - predict, g - ground truth
        p_indices = [i for i, c in enumerate(pred_classes) if int(c) == target_cls]
        g_indices = [j for j, c in enumerate(gt_classes) if int(c) == target_cls]

        # coordinate of rcaing bounding boxes for the current class
        sub_preds = pred_boxes[p_indices] if len(p_indices) > 0 else []
        sub_gts = gt_boxes[g_indices] if len(g_indices) > 0 else []

        gt_matched = [False] * len(sub_gts)

        for sub_p_idx, p_box in enumerate(sub_preds):
            # מוצאים את האינדקס המקורי במערך הכללי של pred_boxes
            orig_p_idx = p_indices[sub_p_idx]

            best_iou = -1
            best_gt_idx = -1

            for j, g_box in enumerate(sub_gts):
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if MODE =='CLASSIC':
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    image_metrics[target_cls]["tp"] += 1
                    gt_matched[best_gt_idx] = True
                    pred_statuses[orig_p_idx] = 'tp'  # מסמנים כהצלחה לטובת הציור
                else:
                    image_metrics[target_cls]["fp"] += 1
                    pred_statuses[orig_p_idx] = 'fp'  # מסמנים כטעות/כפילות לטובת הציור
            elif MODE =='ADVANCED':
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    image_metrics[target_cls]["tp"] += 1
                    gt_matched[best_gt_idx] = True
                    pred_statuses[orig_p_idx] = 'tp'
                elif best_iou >= iou_threshold and gt_matched[best_gt_idx]:
                    # 2. חפיפה טובה אבל הדג כבר תפוס -> לא עושים כלום!
                    # התיבה לא מקבלת TP אבל גם לא נענשת ב-FP.
                    # לטובת הציור הגראפי, נסמן אותה למשל כ-'duplicate' או 'tp' כדי שלא תהיה אדומה
                    pred_statuses[orig_p_idx] = 'tp'
                else:
                    # 3. best_iou < iou_threshold -> חפיפה נמוכה או אין GT בכלל -> FP
                    image_metrics[target_cls]["fp"] += 1
                    pred_statuses[orig_p_idx] = 'fp'


        image_metrics[target_cls]["fn"] = len(sub_gts) - sum(gt_matched)

    return image_metrics, pred_statuses

# General purpose funtion:

def calculate_iou(box1, box2):
    """
    Calculates the IoU (Intersection over Union) between two bounding boxes
    in [x_min, y_min, x_max, y_max] format.
    """

    # 1. Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0]) # most right
    y_top = max(box1[1], box2[1]) # most bottom
    x_right = min(box1[2], box2[2]) # most left
    y_bottom = min(box1[3], box2[3]) # most top

    # no intersection at all
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 2. Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 3. Calculate individual box areas
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 4. Calculate the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Return the ratio (IoU)
    return intersection_area / union_area

def class_agnostic_nms(boxes, confs, iou_thr=NMS_IOU):
    """
    Class-agnostic NMS: Sorts boxes by confidence in descending order, keeps the highest,
    and removes any other box (regardless of class) that overlaps with it above the IoU threshold.
    Repeats for the remaining boxes. Returns the indices of the kept boxes.
    """
    n = len(boxes)
    if n == 0:
        return []

    # Sort indices by confidence in descending order
    order = list(np.argsort(confs)[::-1])
    keep = []
    while order:
        i = order.pop(0) # Get the remaining box with the highest confidence
        keep.append(i)

        # Keep only the boxes that do not overlap with it above the threshold
        order = [j for j in order if calculate_iou(boxes[i], boxes[j]) <= iou_thr]
    return keep

if __name__ == "__main__":
    if ANALYZE_PREDICT == True:
        current_project_dir = os.path.dirname(os.path.abspath(__file__))
        runs_output_dir = os.path.join(current_project_dir, "runs")
        val_images_dir = "datasets/data/valid/images"
        val_labels_dir = "datasets/data/valid/labels"

        model = YOLO("version6.pt", task="detect")

        lb_params = {}  # basename(ללא סיומת) -> (ratio, dw, dh, w0, h0)

        if USE_VAL_PREPROCESS:
            # שכפול ה-preprocess של model.val(): letterbox מלבני ל-VAL_IMGSZ (תוכן native + מסגרת אפורה).
            # האצה: כותבים את התמונות ה-letterboxed פעם אחת לתיקייה זמנית ומריצים predict על התיקייה
            # (אותו dataloader מהיר כמו predict רגיל), במקום להעביר רשימת arrays בזיכרון.
            lb = LetterBox((VAL_IMGSZ, VAL_IMGSZ), auto=False, scaleup=False)
            lb_input_dir = os.path.join(current_project_dir, "_val_preprocess_input")
            os.makedirs(lb_input_dir, exist_ok=True)
            for old in glob.glob(os.path.join(lb_input_dir, "*")):
                os.remove(old)
            for f in sorted(glob.glob(os.path.join(val_images_dir, "*.*"))):
                im = cv2.imread(f)
                h0, w0 = im.shape[:2]
                ratio = min(VAL_IMGSZ / w0, VAL_IMGSZ / h0, 1.0)
                nw, nh = round(w0 * ratio), round(h0 * ratio)
                dw, dh = (VAL_IMGSZ - nw) / 2, (VAL_IMGSZ - nh) / 2
                stem = os.path.splitext(os.path.basename(f))[0]
                # שומרים כ-PNG (ללא דחיסה) כדי לא לפגוע בפיקסלים — JPEG חוזר היה משנה את התוצאות
                cv2.imwrite(os.path.join(lb_input_dir, stem + ".png"), lb(image=im))
                lb_params[stem] = (ratio, dw, dh, w0, h0, f)
            predict_input = lb_input_dir
            predict_imgsz = VAL_IMGSZ
        else:
            predict_input = val_images_dir
            predict_imgsz = 640

        results = model.predict(predict_input,
                                agnostic_nms=False,
                                conf=CONF,
                                iou=0.45,
                                imgsz=predict_imgsz,
                                save=True,
                                save_txt=True,
                                save_conf=True,
                                project=runs_output_dir)

        # שלב 2 (אם דלוק): טוענים פעם אחת את מסווג ה-ConvNeXt + הטרנספורם + מיפוי התמונות המקוריות
        if ANALYSE_2_STEP:
            import torch
            import torchvision.models as tvm
            cn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            convnext = load_convnext_classifier(CKPT_PATH, cn_device)
            cn_transform = tvm.ConvNeXt_Tiny_Weights.DEFAULT.transforms()
            orig_lookup = build_orig_lookup(ORIG_IMAGES_DIR)
            CN_TO_YOLO = {0: None, 1: 0, 2: 1}   # background->drop, fish->0, partial_fish->1
            print(f"[2-STEP] ConvNeXt נטען (device={cn_device}), {len(orig_lookup)} תמונות מקור מופו")

        global_metrics = {
            0: {"tp": 0, "fp": 0, "fn": 0},
            1: {"tp": 0, "fp": 0, "fn": 0}
        }

        # מוני ספירה (MAE) בסף ה-CONF הנוכחי
        count_n_images = 0
        count_abs = {0: 0.0, 1: 0.0, "total": 0.0}   # סכום |נספרו - GT| לתמונה
        count_gt = {0: 0, 1: 0}                        # סכום GT לכל מחלקה (לחישוב ממוצע לתמונה)

        class_names = {0: "Fish", 1: "Partial"}
        for res in results:
            img_name = os.path.splitext(os.path.basename(res.path))[0]
            # במצב val-preprocess מציירים ומתייגים על התמונה המקורית (640), לא על ה-letterbox (672)
            if USE_VAL_PREPROCESS:
                img_path = lb_params[img_name][5]
            else:
                img_path = res.path
            gt_label_path = os.path.join(val_labels_dir, f"{img_name}.txt")

            if not os.path.exists(gt_label_path):
                print(f"❌ [שגיאה] לא נמצא קובץ תיוג עבור התמונה: {img_name}")
                continue

            print(f"\nמעבד את תמונה: {img_name}")

            raw_boxes = res.boxes.xyxy.cpu().numpy()
            raw_classes = res.boxes.cls.cpu().numpy()
            raw_confs = res.boxes.conf.cpu().numpy()

            if USE_VAL_PREPROCESS:
                # מחזירים את הקופסאות מקואורדינטות ה-letterbox (VAL_IMGSZ) לקואורדינטות המקוריות
                ratio, dw, dh, w0, h0, _ = lb_params[img_name]
                raw_boxes = raw_boxes.copy()
                raw_boxes[:, [0, 2]] = (raw_boxes[:, [0, 2]] - dw) / ratio
                raw_boxes[:, [1, 3]] = (raw_boxes[:, [1, 3]] - dh) / ratio
                orig_w, orig_h = w0, h0  # הגודל המקורי האמיתי, כדי שה-GT ייושב באותו מרחב
            else:
                orig_h, orig_w = res.orig_shape

            # class-agnostic NMS אחרי ה-detection (רק במצב חד-שלבי): מסיר כפילויות חוצות-מחלקה.
            # בדו-שלבי ה-NMS עובר לאחרי הקלאסיפייר (ראה למטה), כדי לדה-דופ לפי ההחלטות המתוקנות.
            if USE_NMS and not ANALYSE_2_STEP:
                keep = class_agnostic_nms(raw_boxes, raw_confs, NMS_IOU)
                raw_boxes = raw_boxes[keep]
                raw_classes = raw_classes[keep]
                raw_confs = raw_confs[keep]

            # שלב 2: חיתוך כל גילוי מהתמונה המקורית -> ConvNeXt מסווג מחדש -> שינוי שיוך / השלכת background
            if ANALYSE_2_STEP:
                orig_path = orig_lookup.get(_original_stem(img_name))
                orig_im = cv2.imread(orig_path) if orig_path else None
                if orig_im is None:
                    print(f"   ⚠️ [2-STEP] לא נמצאה תמונת מקור ל-{img_name} — משאיר חיזויי YOLO כמו שהם")
                else:
                    oh, ow = orig_im.shape[:2]
                    crops, idxs = [], []
                    for i in range(len(raw_boxes)):
                        # raw_boxes ב-640px; מנרמלים (חלוקה ב-orig_w/orig_h=640) וממירים לממדי המקור
                        x1 = raw_boxes[i, 0] / orig_w * ow
                        y1 = raw_boxes[i, 1] / orig_h * oh
                        x2 = raw_boxes[i, 2] / orig_w * ow
                        y2 = raw_boxes[i, 3] / orig_h * oh
                        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, ow, oh, CROP_PADDING)
                        crop = orig_im[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        crops.append(crop)
                        idxs.append(i)
                    preds = classify_crops(convnext, cn_transform, crops, cn_device)
                    yolo_confs = raw_confs  # שומרים את ה-confidence של YOLO — זה מה שמסננים לפיו
                    nb, ncl, ncf = [], [], []
                    for i, (cn_cls, _cn_conf) in zip(idxs, preds):
                        ycls = CN_TO_YOLO[cn_cls]
                        if ycls is None:      # background -> משליכים את הגילוי
                            continue
                        # ConvNeXt קובע רק את המחלקה; ה-confidence נשאר של YOLO (לא מסננים מחדש)
                        nb.append(raw_boxes[i]); ncl.append(ycls); ncf.append(yolo_confs[i])
                    raw_boxes = np.array(nb).reshape(-1, 4)
                    raw_classes = np.array(ncl, dtype=float)
                    raw_confs = np.array(ncf, dtype=float)

                # NMS אחרי הקלאסיפייר (בדו-שלבי תמיד אחרי, אם USE_NMS): דה-דופ לפי הזיהויים
                # המתוקנים (אחרי תיקון מחלקות והשלכת background), לפי ה-confidence של YOLO.
                if USE_NMS and len(raw_boxes):
                    keep = class_agnostic_nms(raw_boxes, raw_confs, NMS_IOU)
                    raw_boxes = raw_boxes[keep]
                    raw_classes = raw_classes[keep]
                    raw_confs = raw_confs[keep]

            sort_indices = np.argsort(raw_confs)[::-1]

            pred_boxes = raw_boxes[sort_indices]
            pred_classes = raw_classes[sort_indices]


            gt_boxes = []
            gt_classes = []

            with open(gt_label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        x_min = (cx - w / 2) * orig_w
                        y_min = (cy - h / 2) * orig_h
                        x_max = (cx + w / 2) * orig_w
                        y_max = (cy + h / 2) * orig_h
                        gt_boxes.append([x_min, y_min, x_max, y_max])
                        gt_classes.append(cls_id)

            gt_boxes = np.array(gt_boxes) if len(gt_boxes) > 0 else np.empty((0, 4))
            gt_classes = np.array(gt_classes) if len(gt_classes) > 0 else np.empty((0,))

            # קריאה לפונקציה המעודכנת (שמחזירה גם את סטטוס הניבויים)
            img_results, pred_statuses = evaluate_class_specific(pred_boxes, pred_classes, gt_boxes, gt_classes,
                                                                 iou_threshold=0.5)

            # הדפסת המצב הטקסטואלי בטרמינל
            print(f"   🐟 FISH:         TP={img_results[0]['tp']}, FP={img_results[0]['fp']}, FN={img_results[0]['fn']}")
            print(f"   ✂️ PARTIAL FISH: TP={img_results[1]['tp']}, FP={img_results[1]['fp']}, FN={img_results[1]['fn']}")

            # עדכון המונים הגלובליים
            for c in [0, 1]:
                global_metrics[c]["tp"] += img_results[c]["tp"]
                global_metrics[c]["fp"] += img_results[c]["fp"]
                global_metrics[c]["fn"] += img_results[c]["fn"]

            # ספירה: צבירת שגיאת |נספרו - GT| לתמונה (כולל ולכל מחלקה בנפרד)
            count_n_images += 1
            pf = int((pred_classes == 0).sum()); pp = int((pred_classes == 1).sum())
            gf = int((gt_classes == 0).sum()); gp = int((gt_classes == 1).sum())
            count_abs[0] += abs(pf - gf)
            count_abs[1] += abs(pp - gp)
            count_abs["total"] += abs((pf + pp) - (gf + gp))
            count_gt[0] += gf; count_gt[1] += gp

            # ===================================================================
            # 🎨 חלק גראפי חסין קצוות: מניעת התנגשויות + הגבלת גבולות תמונה מוחלטת
            # ===================================================================
            if GRAPHICAL_DEBUG and img_results[0]["fn"]>0:
                img_to_draw = cv2.imread(img_path)
                img_h, img_w, _ = img_to_draw.shape

                # רשימה שתשמור איפה כבר שמנו טקסט בתמונה הזו: (x_start, x_end, y_center)
                occupied_text_areas = []


                def get_non_overlapping_y(x_start, x_end, desired_y, th, img_h, step=18, side='up'):
                    """
                    פונקציה חכמה שמונעת התנגשויות, ומחליפה כיוון באופן אוטומטי
                    אם הטקסט מנסה לצאת מגבולות התמונה (למעלה או למטה).
                    """
                    current_y = desired_y
                    attempts = 0
                    direction = -1 if side == 'up' else 1  # -1 עולה למעלה, 1 יורד למטה

                    while attempts < 10:
                        overlap = False
                        for (ox1, ox2, oy) in occupied_text_areas:
                            if not (x_end < ox1 or x_start > ox2):  # חפיפה ב-X
                                if abs(current_y - oy) < step:  # קרוב מדי ב-Y
                                    overlap = True
                                    break
                        if not overlap:
                            # בדיקה האם המיקום הנוכחי חורג מגבולות התמונה
                            # הטקסט נכתב מ-current_y ומעלה בגובה th
                            if current_y - th - 2 >= 0 and current_y + 4 <= img_h:
                                break

                        # אם יש חפיפה או חריגה מהמסגרת, נתקדם לקומה הבאה בכיוון שנבחר
                        current_y += direction * step
                        attempts += 1

                    # הגנה אחרונה בהחלט (Clamping) כדי שלא יברח מהמסגרת בשום מצב
                    if current_y - th - 2 < 0:
                        current_y = th + 4
                    elif current_y + 4 > img_h:
                        current_y = img_h - 4

                    occupied_text_areas.append((x_start, x_end, current_y))
                    return current_y


                # 1. ציור קופסאות ה-GT (צבע כחול)
                for j, gt_box in enumerate(gt_boxes):
                    g_cls = int(gt_classes[j])
                    x1, y1, x2, y2 = map(int, gt_box)

                    # הגבלת התיבה עצמה שלא תצא מהתמונה
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

                    cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    label_gt = f"GT:{class_names[g_cls]}"
                    (tw, th), _ = cv2.getTextSize(label_gt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

                    # החלטה על כיוון בסיסי: אם התיבה צמודה לתקרה, נרד למטה לתוך התיבה
                    if y1 < 25:
                        base_y = y1 + th + 5
                        side = 'down'
                    else:
                        base_y = y1 - 5
                        side = 'up'

                    safe_y = get_non_overlapping_y(x1, x1 + tw, base_y, th, img_h, side=side)

                    # ציור קו מנחה אם הטקסט נדחף
                    orig_base_y = y1 + th + 5 if y1 < 25 else y1 - 5
                    if abs(safe_y - orig_base_y) > 5:
                        cv2.line(img_to_draw, (x1 + 5, orig_base_y), (x1 + 5, safe_y), (255, 0, 0), 1)

                    cv2.rectangle(img_to_draw, (x1, safe_y - th - 2), (x1 + tw + 4, safe_y + 2), (50, 0, 0), cv2.FILLED)
                    cv2.putText(img_to_draw, label_gt, (x1 + 2, safe_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1, cv2.LINE_AA)

                # 2. ציור קופסאות המודל (ירוק ל-TP, אדום ל-FP)
                for i, p_box in enumerate(pred_boxes):
                    p_cls = int(pred_classes[i])
                    status = pred_statuses[i]
                    x1, y1, x2, y2 = map(int, p_box)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

                    if status == 'tp':
                        color = (0, 255, 0)
                        label = f"TP:{class_names[p_cls]}"
                    else:
                        color = (0, 0, 255)
                        label = f"FP:{class_names[p_cls]}"

                    cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), color, 2)

                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

                    # החלטה על כיוון בסיסי לניבוי
                    if y1 < 25:
                        base_y = y1 + th + 20  # נמוך יותר מה-GT
                        side = 'down'
                    else:
                        base_y = y1 - 5
                        side = 'up'

                    safe_y = get_non_overlapping_y(x1, x1 + tw, base_y, th, img_h, side=side)

                    orig_base_y = y1 + th + 20 if y1 < 25 else y1 - 5
                    if abs(safe_y - orig_base_y) > 5:
                        cv2.line(img_to_draw, (x1 + tw // 2, orig_base_y), (x1 + tw // 2, safe_y), color, 1)
                        cv2.circle(img_to_draw, (x1 + tw // 2, orig_base_y), 2, color, -1)

                    cv2.rectangle(img_to_draw, (x1, safe_y - th - 2), (x1 + tw + 4, safe_y + baseline), (0, 0, 0),
                                  cv2.FILLED)
                    cv2.putText(img_to_draw, label, (x1 + 2, safe_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

                # הצגת התמונה
                cv2.imshow("Graphical Debug - Press ANY KEY to continue", img_to_draw)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # ===================================================================
        # 🏆 בלוק סיכום סופי חגיגי מחוץ ללולאה 🏆
        # ===================================================================
        print("\n" + "=" * 60)
        print(f" 📊 דוח סיכום סופי - מוד עבודה: {MODE} 📊")
        print("=" * 60)

        class_labels = {0: "FISH (דג שלם)", 1: "PARTIAL FISH (דג חלקי)"}

        for c in [0, 1]:
            tp = global_metrics[c]["tp"]
            fp = global_metrics[c]["fp"]
            fn = global_metrics[c]["fn"]

            # חישוב המדדים הנגזרים (עם הגנה מפני חלוקה באפס)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n🐟 מחלקה {c}: {class_labels[c]}")
            print(f"   🔹 סך הכל True Positives  (TP): {tp}")
            print(f"   🔹 סך הכל False Positives (FP): {fp}")
            print(f"   🔹 סך הכל False Negatives (FN): {fn}")
            print(f"   ----------------------------------")
            print(f"   📈 Precision (דיוק הזיהוי):      {precision:.4f}")
            print(f"   📉 Recall (אחוז הגילוי):         {recall:.4f}")
            print(f"   🏅 F1-Score (מדד משולב):         {f1_score:.4f}")
            print("-" * 45)

        print("=" * 60)

        # ===================================================================
        # 🔢 דוח ספירה (MAE) בסף ה-CONF הנוכחי
        # ===================================================================
        if count_n_images > 0:
            n = count_n_images
            print(f"\n 🔢 דוח ספירה (conf={CONF}, NMS={'ON' if USE_NMS else 'OFF'})")
            print("-" * 45)
            print(f"   דגים שלמים  (fish)    בממוצע לתמונה (GT): {count_gt[0] / n:.2f}")
            print(f"   דגים חלקיים (partial) בממוצע לתמונה (GT): {count_gt[1] / n:.2f}")
            print(f"   סך הכל אובייקטים        בממוצע לתמונה (GT): {(count_gt[0] + count_gt[1]) / n:.2f}")
            print(f"   ----------------------------------")
            print(f"   📏 MAE ממוצע כולל לתמונה:        {count_abs['total'] / n:.3f}")
            print(f"   📏 MAE ממוצע לדגים שלמים:        {count_abs[0] / n:.3f}")
            print(f"   📏 MAE ממוצע לדגים חלקיים:       {count_abs[1] / n:.3f}")
            print("=" * 60)

# Val
    if ANALYZE_VAL:
        datasets_dir = "C:/Users/ndvam/PycharmProjects/model_b/datasets"
        DATA_YAML = f"{datasets_dir}/data/data.yaml"

        model.val(data=DATA_YAML,
                  agnostic_nms=False,
                  split="val",
                  conf=CONF,
                  iou=0.5,
                  imgsz=640,
                  max_det=1000,
                  device=0,
                  save_txt=True,
                  project=runs_output_dir,
                  plots=True)