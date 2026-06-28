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
ANALYSE_2_STEP = True

# When True: after predicting, run the FULL evaluation (TP/FP/FN, P/R/F1, MAE/MAPE, pixelwise).
# When False: ONLY predict and save annotated images + label .txt files — use this to LABEL images.
RUN_EVALUATION = True

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
USE_NMS = True
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

def evaluate_class_agnostic(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=IOU_THRESHOLD):
    """
    שיוך class-agnostic: חיזוי יכול להתאים ל-GT מכל מחלקה (שיוך לפי IoU בלבד, מהגבוה
    לנמוך) — בדיוק כמו evaluate_class_specific אך *ללא* איפוס זוגות ממחלקות שונות.
    בלבול מחלקות נסלח: דג שזוהה כ"חלקי" וחופף ל-GT של דג שלם נחשב התאמה.

    מחזירה fp, fn — מילונים {0,1}:
      • fp[c] = מספר חיזויים שלא הותאמו לאף GT (גילו רקע), לפי מחלקת *החיזוי*.
      • fn[c] = מספר אובייקטי GT שלא הותאמו לאף חיזוי (פוספסו), לפי מחלקת ה-*GT*.
    מ-fp/fn מחושב ה-MAE ה-class-agnostic לכל מחלקה: |fp[c] - fn[c]|.
    """
    fp = {0: 0, 1: 0}
    fn = {0: 0, 1: 0}
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    # מקרי קצה (מחזירים גם קבוצות שיוך ריקות, לחישוב שטחי FP/FN במעלה הזרם)
    if n_pred == 0:
        for g_cls in gt_classes:
            fn[int(g_cls)] += 1
        return fp, fn, set(), set()
    if n_gt == 0:
        for p_cls in pred_classes:
            fp[int(p_cls)] += 1
        return fp, fn, set(), set()

    # מטריצת IoU בין כל GT (שורות) לכל חיזוי (עמודות) — בלי מסכת correct_class
    iou = np.zeros((n_gt, n_pred))
    for g in range(n_gt):
        for p in range(n_pred):
            iou[g, p] = calculate_iou(pred_boxes[p], gt_boxes[g])

    matched_gt = set()
    matched_pred = set()

    # כל הזוגות מעל הסף, ממוינים מהגבוה לנמוך, ואז ייחוד לפי חיזוי ואז לפי GT (כמו match_predictions)
    cand = np.argwhere(iou >= iou_threshold)  # [gt_idx, pred_idx]
    if len(cand) > 0:
        order = iou[cand[:, 0], cand[:, 1]].argsort()[::-1]
        cand = cand[order]
        cand = cand[np.unique(cand[:, 1], return_index=True)[1]]  # חיזוי יחיד לכל חיזוי
        cand = cand[np.unique(cand[:, 0], return_index=True)[1]]  # GT יחיד לכל GT
        for g_idx, p_idx in cand:
            matched_gt.add(int(g_idx))
            matched_pred.add(int(p_idx))

    # חיזויים שלא שויכו => FP לפי מחלקת החיזוי
    for p in range(n_pred):
        if p not in matched_pred:
            fp[int(pred_classes[p])] += 1
    # GT שלא שויכו => FN לפי מחלקת ה-GT
    for g in range(n_gt):
        if g not in matched_gt:
            fn[int(gt_classes[g])] += 1
    return fp, fn, matched_pred, matched_gt

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

        if not RUN_EVALUATION:
            print(f"\n[PREDICT-ONLY] RUN_EVALUATION=False — בוצעה פרדיקציה בלבד. "
                  f"התמונות המסומנות וקבצי התיוג (.txt) נשמרו תחת: {runs_output_dir}. דילוג על ההערכה.")

        # שלב 2 (אם דלוק): טוענים פעם אחת את מסווג ה-ConvNeXt + הטרנספורם + מיפוי התמונות המקוריות
        if ANALYSE_2_STEP and RUN_EVALUATION:
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
        count_ape = {0: 0.0, 1: 0.0, "total": 0.0}   # סכום |נספרו - GT| / GT לתמונה (ל-MAPE)
        count_ape_n = {0: 0, 1: 0, "total": 0}       # מס' תמונות עם GT>0 (מכנה ה-MAPE, מדלג על חלוקה ב-0)

        # מוני ספירה class-agnostic (שיוך לפי IoU בלבד; בלבול מחלקות נסלח, FP לפי מחלקת החיזוי)
        count_abs_ca = {0: 0.0, 1: 0.0, "total": 0.0}   # סכום |FP - FN| לתמונה (MAE)
        count_ape_ca = {0: 0.0, 1: 0.0, "total": 0.0}   # סכום |FP - FN| / GT לתמונה (MAPE)
        count_ape_ca_n = {0: 0, 1: 0, "total": 0}       # מס' תמונות עם GT>0 (מכנה ה-MAPE)

        # =====================================================================
        # 🧮 MISSED/INVENTED AREA (שיוך IoU, מאוגד על כל הסט) — "כמה שטח של אובייקטים
        #    *שלמים* פוספס/הומצא". מבוסס שיוך class-agnostic ב-IoU≥0.5: אובייקט שהותאם
        #    נחשב מכוסה (תורם 0); רק אובייקט שלא הותאם כלל תורם את *כל* שטח התיבה שלו.
        #      under-count = Σ שטח(GT שלא הותאם) / Σ שטח(GT)   ← שטח אובייקטים שפוספסו לגמרי
        #      over-count  = Σ שטח(חיזוי שלא הותאם) / Σ שטח(GT) ← שטח גילויי-שווא
        #    מאוגד (micro): סוכמים שטחים על כל הסט ואז מחלקים. מוכיח ש"הפספוסים קטנים"
        #    בהשוואה לאחוז הפספוס במספר (recall): הרבה אובייקטים פוספסו אך מעט שטח.
        # =====================================================================
        px_gt_sum = {0: 0.0, 1: 0.0, "total": 0.0}   # סך שטח GT לכל מחלקה
        px_fn_sum = {0: 0.0, 1: 0.0, "total": 0.0}   # סך שטח GT שפוספס (אובייקטים שלא הותאמו)
        px_fp_sum = {0: 0.0, 1: 0.0, "total": 0.0}   # סך שטח חיזויים שלא הותאמו (שווא)

        # =====================================================================
        # 🟩 PIXEL-LEVEL COVERAGE (mask-level, ללא שיוך וללא סף) — "כמה אחוז מ*שטח
        #    הפיקסלים* של הדגים באמת כוסה". מרסטרים את כל תיבות ה-GT ואת כל תיבות
        #    החיזוי למסכות פיקסלים וסופרים חפיפה ממשית. בניגוד ל-🧮, כאן תיבה שהותאמה
        #    אך גדולה/קטנה מדי *נקנסת*: פיקסלים עודפים -> over, חסרים -> miss.
        #      coverage = Σ |GT ∩ pred| / Σ |GT|   ← שטח דגים שכוסה נכון
        #      miss = 1 - coverage                 ← שטח שלא כוסה (פספוס + תת-גודל תיבות)
        #      over = Σ |pred \ GT| / Σ |GT|        ← שטח חיזוי מחוץ לכל דג
        #    מודד נאמנות-שטח כוללת (תומך ב"המודל מכסה X% משטח הדגים").
        # =====================================================================
        cov_tp = {0: 0, 1: 0, "total": 0}   # פיקסלי GT שכוסו ע"י חיזוי כלשהו
        cov_gt = {0: 0, 1: 0, "total": 0}   # סך פיקסלי GT
        cov_fp = 0                          # פיקסלי חיזוי מחוץ לכל GT

        class_names = {0: "Fish", 1: "Partial"}
        for res in (results if RUN_EVALUATION else []):
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
            # MAPE: |נספרו - GT| / GT, רק בתמונות עם GT>0 (מניעת חלוקה באפס)
            if gf > 0:
                count_ape[0] += abs(pf - gf) / gf;            count_ape_n[0] += 1
            if gp > 0:
                count_ape[1] += abs(pp - gp) / gp;            count_ape_n[1] += 1
            if (gf + gp) > 0:
                count_ape["total"] += abs((pf + pp) - (gf + gp)) / (gf + gp); count_ape_n["total"] += 1

            # ספירה class-agnostic: שיוך לפי IoU בלבד -> MAE/MAPE לכל מחלקה = |FP - FN|
            ca_fp, ca_fn, ca_matched_pred, ca_matched_gt = evaluate_class_agnostic(
                pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5)
            err_ca_fish = abs(ca_fp[0] - ca_fn[0])
            err_ca_part = abs(ca_fp[1] - ca_fn[1])
            err_ca_total = abs((ca_fp[0] + ca_fp[1]) - (ca_fn[0] + ca_fn[1]))
            count_abs_ca[0] += err_ca_fish
            count_abs_ca[1] += err_ca_part
            count_abs_ca["total"] += err_ca_total
            if gf > 0:
                count_ape_ca[0] += err_ca_fish / gf;            count_ape_ca_n[0] += 1
            if gp > 0:
                count_ape_ca[1] += err_ca_part / gp;            count_ape_ca_n[1] += 1
            if (gf + gp) > 0:
                count_ape_ca["total"] += err_ca_total / (gf + gp); count_ape_ca_n["total"] += 1

            # 🧮 שטח אובייקטים שפוספסו/הומצאו לגמרי (שיוך class-agnostic): שטח GT שלא הותאם
            # לאף חיזוי = FN-area (פוספס), שטח חיזוי שלא הותאם לאף GT = FP-area (שווא).
            gt_area_c = {0: 0.0, 1: 0.0}; fn_area_c = {0: 0.0, 1: 0.0}; fp_area_c = {0: 0.0, 1: 0.0}
            for g in range(len(gt_boxes)):
                a = max(0.0, float(gt_boxes[g][2] - gt_boxes[g][0])) * max(0.0, float(gt_boxes[g][3] - gt_boxes[g][1]))
                c = int(gt_classes[g]); gt_area_c[c] += a
                if g not in ca_matched_gt:
                    fn_area_c[c] += a
            for p in range(len(pred_boxes)):
                if p not in ca_matched_pred:
                    a = max(0.0, float(pred_boxes[p][2] - pred_boxes[p][0])) * max(0.0, float(pred_boxes[p][3] - pred_boxes[p][1]))
                    fp_area_c[int(pred_classes[p])] += a
            ga_tot = gt_area_c[0] + gt_area_c[1]

            # 🧮 צבירה מאוגדת על כל הסט (Σ שטח)
            for c in (0, 1):
                px_gt_sum[c] += gt_area_c[c]; px_fn_sum[c] += fn_area_c[c]; px_fp_sum[c] += fp_area_c[c]
            px_gt_sum["total"] += ga_tot
            px_fn_sum["total"] += fn_area_c[0] + fn_area_c[1]
            px_fp_sum["total"] += fp_area_c[0] + fp_area_c[1]

            # כיסוי פיקסלי ברמת מסכה: מציירים את כל תיבות ה-GT (לפי מחלקה) ואת כל תיבות החיזוי
            # (class-agnostic) על רשת הפיקסלים, וסופרים חיתוך/חוסר/עודף — ללא סף בינארי. כך תיבה
            # מותאמת אך גדולה/קטנה מדי כבר *לא* נחשבת מושלמת: העודף נכנס ל-FP, החוסר ל-FN.
            Hh, Ww = int(orig_h), int(orig_w)
            pred_mask = np.zeros((Hh, Ww), dtype=bool)
            for p in range(len(pred_boxes)):
                x1 = max(0, int(round(pred_boxes[p][0]))); y1 = max(0, int(round(pred_boxes[p][1])))
                x2 = min(Ww, int(round(pred_boxes[p][2]))); y2 = min(Hh, int(round(pred_boxes[p][3])))
                if x2 > x1 and y2 > y1:
                    pred_mask[y1:y2, x1:x2] = True
            gt_mask_all = np.zeros((Hh, Ww), dtype=bool)
            for c in (0, 1):
                gt_mask_c = np.zeros((Hh, Ww), dtype=bool)
                for g in range(len(gt_boxes)):
                    if int(gt_classes[g]) != c:
                        continue
                    x1 = max(0, int(round(gt_boxes[g][0]))); y1 = max(0, int(round(gt_boxes[g][1])))
                    x2 = min(Ww, int(round(gt_boxes[g][2]))); y2 = min(Hh, int(round(gt_boxes[g][3])))
                    if x2 > x1 and y2 > y1:
                        gt_mask_c[y1:y2, x1:x2] = True
                gt_n = int(gt_mask_c.sum())
                tp_n = int((gt_mask_c & pred_mask).sum())
                cov_gt[c] += gt_n; cov_tp[c] += tp_n
                cov_gt["total"] += gt_n; cov_tp["total"] += tp_n
                gt_mask_all |= gt_mask_c
            cov_fp += int((pred_mask & ~gt_mask_all).sum())  # פיקסלי חיזוי מחוץ לכל GT

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
        if RUN_EVALUATION:
            print("\n" + "=" * 60)
            print(f" 📊 דוח סיכום סופי - מוד עבודה: {MODE} 📊")
            print("=" * 60)

            class_labels = {0: "FISH (דג שלם)", 1: "PARTIAL FISH (דג חלקי)"}

            per_class_pr = {}  # c -> (precision, recall, f1) לחישוב שורת "all" המשולבת
            for c in [0, 1]:
                tp = global_metrics[c]["tp"]
                fp = global_metrics[c]["fp"]
                fn = global_metrics[c]["fn"]

                # חישוב המדדים הנגזרים (עם הגנה מפני חלוקה באפס)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                per_class_pr[c] = (precision, recall, f1_score)

                print(f"\n🐟 מחלקה {c}: {class_labels[c]}")
                print(f"   🔹 סך הכל True Positives  (TP): {tp}")
                print(f"   🔹 סך הכל False Positives (FP): {fp}")
                print(f"   🔹 סך הכל False Negatives (FN): {fn}")
                print(f"   ----------------------------------")
                print(f"   📈 Precision (דיוק הזיהוי):      {precision:.4f}")
                print(f"   📉 Recall (אחוז הגילוי):         {recall:.4f}")
                print(f"   🏅 F1-Score (מדד משולב):         {f1_score:.4f}")
                print("-" * 45)

            # שורת "all classes" משולבת — macro-average על שתי המחלקות, כמו Ultralytics (שורת 'all'
            # ב-model.val(): mean precision / mean recall על המחלקות; F1 הכולל = ממוצע ה-F1 הפר-מחלקתי,
            # כמו עקומת ה-F1 ל-'all classes').
            p_all = (per_class_pr[0][0] + per_class_pr[1][0]) / 2
            r_all = (per_class_pr[0][1] + per_class_pr[1][1]) / 2
            f1_all = (per_class_pr[0][2] + per_class_pr[1][2]) / 2
            print(f"\n🎯 ALL CLASSES (כל המחלקות יחד, macro-average כמו Ultralytics)")
            print(f"   📈 Precision (ממוצע מחלקות):     {p_all:.4f}")
            print(f"   📉 Recall (ממוצע מחלקות):        {r_all:.4f}")
            print(f"   🏅 F1-Score (ממוצע מחלקות):      {f1_all:.4f}")
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
            print(f"   ----------------------------------")
            mape_total = (count_ape['total'] / count_ape_n['total'] * 100) if count_ape_n['total'] > 0 else float('nan')
            mape_fish  = (count_ape[0] / count_ape_n[0] * 100) if count_ape_n[0] > 0 else float('nan')
            mape_part  = (count_ape[1] / count_ape_n[1] * 100) if count_ape_n[1] > 0 else float('nan')
            print(f"   📐 MAPE כולל לתמונה:             {mape_total:.2f}%  (על {count_ape_n['total']} תמונות עם GT>0)")
            print(f"   📐 MAPE לדגים שלמים:             {mape_fish:.2f}%  (על {count_ape_n[0]} תמונות עם GT>0)")
            print(f"   📐 MAPE לדגים חלקיים:            {mape_part:.2f}%  (על {count_ape_n[1]} תמונות עם GT>0)")
            print("=" * 60)

            # ===================================================================
            # 🔀 דוח ספירה class-agnostic — שיוך לפי IoU בלבד (בלבול מחלקות נסלח)
            #    MAE/MAPE לכל מחלקה = |FP - FN|; שגיאה רק מ-גילוי רקע (FP) או פספוס (FN)
            # ===================================================================
            print(f"\n 🔀 דוח ספירה class-agnostic (שיוך IoU≥0.5, conf={CONF}, NMS={'ON' if USE_NMS else 'OFF'})")
            print("-" * 45)
            print(f"   📏 MAE לדגים שלמים  (agnostic): {count_abs_ca[0] / n:.3f}")
            print(f"   📏 MAE לדגים חלקיים (agnostic): {count_abs_ca[1] / n:.3f}")
            print(f"   📏 MAE כולל          (agnostic): {count_abs_ca['total'] / n:.3f}")
            print(f"   ----------------------------------")
            mape_ca_fish  = (count_ape_ca[0] / count_ape_ca_n[0] * 100) if count_ape_ca_n[0] > 0 else float('nan')
            mape_ca_part  = (count_ape_ca[1] / count_ape_ca_n[1] * 100) if count_ape_ca_n[1] > 0 else float('nan')
            mape_ca_total = (count_ape_ca['total'] / count_ape_ca_n['total'] * 100) if count_ape_ca_n['total'] > 0 else float('nan')
            print(f"   📐 MAPE לדגים שלמים  (agnostic): {mape_ca_fish:.2f}%  (על {count_ape_ca_n[0]} תמונות עם GT>0)")
            print(f"   📐 MAPE לדגים חלקיים (agnostic): {mape_ca_part:.2f}%  (על {count_ape_ca_n[1]} תמונות עם GT>0)")
            print(f"   📐 MAPE כולל          (agnostic): {mape_ca_total:.2f}%  (על {count_ape_ca_n['total']} תמונות עם GT>0)")
            print("=" * 60)

            # ===================================================================
            # 🧮 שבר שטח מאוגד על כל הסט יחד (micro-average) — פספוס ועודף בנפרד, ללא קיזוז.
            #    under-count = Σ שטח_FN / Σ שטח_GT ; over-count = Σ שטח_FP / Σ שטח_GT
            # ===================================================================
            miss_fish = (px_fn_sum[0] / px_gt_sum[0] * 100) if px_gt_sum[0] > 0 else float('nan')
            miss_part = (px_fn_sum[1] / px_gt_sum[1] * 100) if px_gt_sum[1] > 0 else float('nan')
            miss_tot  = (px_fn_sum['total'] / px_gt_sum['total'] * 100) if px_gt_sum['total'] > 0 else float('nan')
            over_fish = (px_fp_sum[0] / px_gt_sum[0] * 100) if px_gt_sum[0] > 0 else float('nan')
            over_part = (px_fp_sum[1] / px_gt_sum[1] * 100) if px_gt_sum[1] > 0 else float('nan')
            over_tot  = (px_fp_sum['total'] / px_gt_sum['total'] * 100) if px_gt_sum['total'] > 0 else float('nan')
            print(f"\n 🧮 שטח אובייקטים שפוספסו/הומצאו לגמרי (שיוך IoU≥0.5, מאוגד, conf={CONF})")
            print("-" * 45)
            print(f"   [שטח התיבה של אובייקטים שלא הותאמו כלל. אובייקט שהותאם=מכוסה. מוכיח שהפספוסים קטנים.]")
            print(f"   🔻 שטח דגים שפוספס (under-count) מתוך סך שטח ה-GT:")
            print(f"        דגים שלמים: {miss_fish:.2f}%   דגים חלקיים: {miss_part:.2f}%   כולל: {miss_tot:.2f}%")
            print(f"   🔺 שטח רקע שהומצא (over-count) מתוך סך שטח ה-GT:")
            print(f"        דגים שלמים: {over_fish:.2f}%   דגים חלקיים: {over_part:.2f}%   כולל: {over_tot:.2f}%")
            print("=" * 60)

            # ===================================================================
            # 🟩 כיסוי שטח פיקסלי (mask-level, ללא סף) — תומך בטענה "המודל מכסה נכון X% משטח הדגים".
            #    coverage = ΣTP_px / Σשטח_GT ; חוסר = 100%-coverage ; עודף = פיקסלי חיזוי מחוץ ל-GT.
            # ===================================================================
            cov_f = (cov_tp[0] / cov_gt[0] * 100) if cov_gt[0] > 0 else float('nan')
            cov_p = (cov_tp[1] / cov_gt[1] * 100) if cov_gt[1] > 0 else float('nan')
            cov_t = (cov_tp['total'] / cov_gt['total'] * 100) if cov_gt['total'] > 0 else float('nan')
            fp_pct = (cov_fp / cov_gt['total'] * 100) if cov_gt['total'] > 0 else float('nan')
            print(f"\n 🟩 כיסוי שטח פיקסלי (mask-level, ללא סף, class-agnostic, conf={CONF})")
            print("-" * 45)
            print(f"   [חפיפת פיקסלים ממשית. תיבה לא-מדויקת נקנסת. מודד נאמנות-שטח כוללת (X% משטח הדגים).]")
            print(f"   ✅ שטח דגים שכוסה נכון (coverage = TP/GT):")
            print(f"        דגים שלמים: {cov_f:.2f}%   דגים חלקיים: {cov_p:.2f}%   כולל: {cov_t:.2f}%")
            print(f"   🔻 שטח דגים שלא כוסה (חוסר): כולל: {100-cov_t:.2f}%")
            print(f"   🔺 שטח חיזוי מחוץ לכל דג (עודף) מתוך שטח ה-GT: {fp_pct:.2f}%")
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