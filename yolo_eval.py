import os

# ===================================================================================
# --- פתרון שגיאת OMP ---
# מגדיר למערכת להתעלם מכפילות של ספריות OpenMP שנטענות על ידי PyTorch ו-Matplotlib
# יש לשים את זה לפני ייבוא של numpy או torch/YOLO
# ===================================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2
from ultralytics import YOLO

# ===================================================================================
# 1. Setup paths and workspace
# ===================================================================================
project_root = r'C:\Users\ndvam\PycharmProjects\model_b'
images_dir = r'C:\Users\ndvam\PycharmProjects\model_b\datasets\data\valid\images'
labels_dir = r'C:\Users\ndvam\PycharmProjects\model_b\datasets\data\valid\labels'

# ===================================================================================
# 2. Define operational thresholds
# ===================================================================================
CONF_THRESHOLD = 0.46
MATCH_IOU_THRESHOLD = 0.5  # סף חפיפה להכרזה על זיהוי מוצלח (True Positive)
NMS_IOU_THRESHOLD = 0.7  # סף NMS בתוך המודל, מונע מחיקת דגים צמודים

# Flags
PER_CLASS_LOCK = True
AGNOSTIC_NMS = False

# ===================================================================================
# 3. Load model
# ===================================================================================
model = YOLO('model.pt')
names = model.names
num_classes = len(names)

confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)


def calculate_iou(box1, box2):
    """מחשב את אחוז החפיפה (IoU) בין שתי קופסאות"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# ===================================================================================
# 4. Process single image
# ===================================================================================
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
if not image_files:
    raise FileNotFoundError("לא נמצאו תמונות בתיקייה!")

target_image_name = image_files[0]
target_image_path = os.path.join(images_dir, target_image_name)

base_name = os.path.splitext(target_image_name)[0]
label_path = os.path.join(labels_dir, base_name + '.txt')

print(f"\n{'=' * 50}")
print(f" מנתח את התמונה: {target_image_name} ")
print(f"{'=' * 50}\n")

# --- אינפרנס ---
results = model.predict(
    source=target_image_path,
    conf=CONF_THRESHOLD,
    iou=NMS_IOU_THRESHOLD,
    device=0,
    verbose=False,
    agnostic_nms=AGNOSTIC_NMS,
    imgsz=640,
    half=False,
    max_det=1000
)
result = results[0]

# --- חילוץ תחזיות ---
if result.boxes is not None and len(result.boxes) > 0:
    pred_boxes = result.boxes.data.cpu().numpy()
    pred_boxes = pred_boxes[np.argsort(-pred_boxes[:, 4])]
else:
    pred_boxes = []

print(f"--> המודל מצא {len(pred_boxes)} תחזיות (Predictions).")

# --- חילוץ GT ---
gt_boxes = []
if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
    gt_data = np.loadtxt(label_path).reshape(-1, 5)
    # אם יש רק אובייקט אחד, הפורמט יכול להיות שורה אחת ולא מטריצה הדו-ממדית. נטפל בזה:
    if gt_data.ndim == 1:
        gt_data = np.array([gt_data])

    h, w = result.orig_shape
    for box in gt_data:
        cls, xc, yc, bw, bh = box
        gt_boxes.append([
            int(cls),
            (xc - bw / 2) * w, (yc - bh / 2) * h,
            (xc + bw / 2) * w, (yc + bh / 2) * h
        ])

print(f"--> בקובץ התיוג יש {len(gt_boxes)} אובייקטים אמיתיים (Ground Truths).\n")
print("-" * 50)

# ===================================================================================
# 5. Matching Logic
# ===================================================================================
matched_gts = set()

for p_idx, pred in enumerate(pred_boxes):
    p_box = pred[0:4]
    p_conf = pred[4]
    p_cls = int(pred[5])

    print(f"בודק תחזית {p_idx + 1}: מחלקה [{names[p_cls]}], ביטחון: {p_conf:.2f}")

    best_iou = 0
    best_gt_idx = -1

    for gt_idx, gt in enumerate(gt_boxes):
        g_cls = gt[0]

        if g_cls != p_cls:
            continue

        if PER_CLASS_LOCK:
            if (gt_idx, p_cls) in matched_gts:
                continue
        else:
            if gt_idx in matched_gts:
                continue

        g_box = gt[1:5]
        iou = calculate_iou(p_box, g_box)

        if iou > best_iou:
            best_iou = iou
            best_gt_idx = gt_idx

    if best_iou >= MATCH_IOU_THRESHOLD and best_gt_idx != -1:
        g_cls = gt_boxes[best_gt_idx][0]
        confusion_matrix[g_cls, p_cls] += 1

        if PER_CLASS_LOCK:
            matched_gts.add((best_gt_idx, p_cls))
        else:
            matched_gts.add(best_gt_idx)

        print(f"    [+] הצלחה (True Positive)! התאים ל-GT מספר {best_gt_idx} עם חפיפה של {best_iou:.2f}\n")
    else:
        confusion_matrix[num_classes, p_cls] += 1
        if best_gt_idx != -1:
            print(
                f"    [-] זיהוי שווא (False Positive). החפיפה הגבוהה ביותר הייתה {best_iou:.2f} (נמוך מ-{MATCH_IOU_THRESHOLD})\n")
        else:
            print(f"    [-] זיהוי שווא (False Positive). התחזית סומנה על רקע ללא GT תואם.\n")

print("-" * 50)
for gt_idx, gt in enumerate(gt_boxes):
    g_cls = gt[0]
    is_missed = False

    if PER_CLASS_LOCK:
        if (gt_idx, g_cls) not in matched_gts:
            is_missed = True
    else:
        if gt_idx not in matched_gts:
            is_missed = True

    if is_missed:
        confusion_matrix[g_cls, num_classes] += 1
        print(f"[!] המודל פספס את GT מספר {gt_idx} ממחלקה [{names[g_cls]}] (False Negative).")

# ===================================================================================
# 6. Print Confusion Matrix (Moved BEFORE Visualization)
# ===================================================================================
print("\n" + "=" * 60)
print(f" מטריצת בלבול עבור התמונה: {target_image_name} ")
print("=" * 60)

header = f"{'True \ Pred':<15}" + "".join([f"{names[idx]:<15}" for idx in range(num_classes)]) + f"{'background':<15}"
print(header)
print("-" * len(header))

for idx in range(num_classes + 1):
    row_label = names[idx] if idx < num_classes else "background"
    row_values = "".join([f"{int(confusion_matrix[idx][jdx]):<15}" for jdx in range(num_classes + 1)])
    print(f"{row_label:<15}{row_values}")

print("=" * 60)

# ===================================================================================
# 7. Interactive Visualization (OpenCV)
# ===================================================================================
# קריאת התמונה המקורית (כבר בפורמט BGR שמתאים ל-OpenCV)
orig_img = cv2.imread(target_image_path)

# הגדרת צבעים למחלקות בפורמט BGR
gt_colors = [(0, 255, 0), (0, 165, 255)]  # GT: ירוק לדג, כתום לדג חלקי
pred_colors = [(0, 0, 255), (255, 0, 255)]  # Preds: אדום לדג, מג'נטה לדג חלקי


# --- פונקציית עזר לציור קופסה עם רקע חצי-שקוף לטקסט ---
def draw_box_with_label(image, box, label, color, is_gt=False):
    x1, y1, x2, y2 = map(int, box)

    # ציור הקופסה עצמה
    thickness = 2 if is_gt else 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # חישוב גודל הטקסט
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    offset_y = 5 if is_gt else -5
    offset_x = 5 if is_gt else -5

    if is_gt:
        text_y = y2 + text_height + offset_y
        rect_y1 = y2 + offset_y
        rect_y2 = y2 + text_height + offset_y + baseline
    else:
        text_y = y1 - offset_y
        rect_y1 = y1 - text_height - offset_y - baseline
        rect_y2 = y1 - offset_y

    text_x = x1 + offset_x

    # שמירה בגבולות
    if rect_y1 < 0:
        rect_y1 = 0
        rect_y2 = text_height + baseline
        text_y = text_height
    if rect_y2 > image.shape[0]:
        rect_y2 = image.shape[0]
        rect_y1 = image.shape[0] - text_height - baseline
        text_y = image.shape[0] - baseline

    cv2.rectangle(image, (text_x, rect_y1), (text_x + text_width, rect_y2), color, cv2.FILLED)
    text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
    cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)


# משתני מצב לשליטה בתצוגה נפרדת לכל מחלקה
show_gt = True
show_pred_fish = True  # מחלקה 0
show_pred_partial = True  # מחלקה 1

window_name = f"Debug View: {target_image_name}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\n" + "=" * 50)
print(" חלון תצוגה נפתח. השתמש במקשים הבאים בתוך החלון:")
print("  'g' - הצג/הסתר Ground Truth (GT)")
print("  'f' - הצג/הסתר תחזיות Fish (אדום)")
print("  'p' - הצג/הסתר תחזיות Partial Fish (מג'נטה)")
print("  'q' או ESC - סגור את החלון")
print("=" * 50 + "\n")

while True:
    # יצירת עותק נקי של התמונה לכל פריים
    display_img = orig_img.copy()

    # ציור GT אם המצב מאפשר
    if show_gt:
        for gt in gt_boxes:
            cls = gt[0]
            color = gt_colors[cls % len(gt_colors)]
            label = f"GT: {names[cls]}"
            draw_box_with_label(display_img, gt[1:5], label, color, is_gt=True)

    # ציור תחזיות מופרד לפי מחלקה (Fish מול Partial Fish)
    for pred in pred_boxes:
        conf = pred[4]
        cls = int(pred[5])

        # סינון לפי הדגלים האינטראקטיביים
        if cls == 0 and not show_pred_fish:
            continue
        if cls == 1 and not show_pred_partial:
            continue

        color = pred_colors[cls % len(pred_colors)]
        label = f"Pred: {names[cls]} ({conf:.2f})"
        draw_box_with_label(display_img, pred[0:4], label, color, is_gt=False)

    # הוספת מקרא נקי על גבי התמונה בשתי שורות
    info_text_1 = f"GT (g): {show_gt} | Fish Pred (f): {show_pred_fish} | Partial Pred (p): {show_pred_partial}"
    info_text_2 = "Press 'q' to Quit"

    cv2.putText(display_img, info_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 3)
    cv2.putText(display_img, info_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.putText(display_img, info_text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(display_img, info_text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.imshow(window_name, display_img)

    # המתנה ללחיצת מקש (1 מילישנייה)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        show_gt = not show_gt
    elif key == ord('f'):  # פילטר לדגים בלבד
        show_pred_fish = not show_pred_fish
    elif key == ord('p'):  # פילטר לדגים חלקיים בלבד
        show_pred_partial = not show_pred_partial
    elif key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()