import os

# ===================================================================================
# --- פתרון שגיאת OMP ---
# מגדיר למערכת להתעלם מכפילות של ספריות OpenMP שנטענות על ידי PyTorch ו-Matplotlib
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

target_image_name = image_files[2]
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

    best_iou = 0
    best_gt_idx = -1

    for gt_idx, gt in enumerate(gt_boxes):
        g_cls = gt[0]
        if g_cls != p_cls: continue
        if PER_CLASS_LOCK:
            if (gt_idx, p_cls) in matched_gts: continue
        else:
            if gt_idx in matched_gts: continue

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
    else:
        confusion_matrix[num_classes, p_cls] += 1

for gt_idx, gt in enumerate(gt_boxes):
    g_cls = gt[0]
    is_missed = True if (PER_CLASS_LOCK and (gt_idx, g_cls) not in matched_gts) or (
                not PER_CLASS_LOCK and gt_idx not in matched_gts) else False
    if is_missed:
        confusion_matrix[g_cls, num_classes] += 1

# ===================================================================================
# 6. Print Separate Tables (Fish & Partial Fish)
# ===================================================================================
print("\n" + "=" * 60)
print(f" סיכום מטריקות לפי מחלקה: {target_image_name} ")
print("=" * 60)

for cls_id in range(num_classes):
    cls_name = names[cls_id]
    TP = confusion_matrix[cls_id, cls_id]
    FP = confusion_matrix[num_classes, cls_id]
    FN = confusion_matrix[cls_id, num_classes]

    print(f"\n--- טבלת נתונים: {cls_name.upper()} ---")
    print(f"  TP (True Positives) : {TP}")
    print(f"  FP (False Positives): {FP}")
    print(f"  FN (False Negatives): {FN}")

print("\n" + "=" * 60 + "\n")

# ===================================================================================
# 7. Interactive Visualization
# ===================================================================================
orig_img = cv2.imread(target_image_path)
gt_colors = [(0, 255, 0), (0, 165, 255)]
pred_colors = [(0, 0, 255), (255, 0, 255)]


def draw_box_with_label(image, box, label, color, is_gt=False):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 5), (x1 + tw, y1), color, cv2.FILLED)
    cv2.putText(image, label, (x1, y1 - 5), font, 0.5, (255, 255, 255), 1)


show_gt = True
show_pred_fish = True
show_pred_partial = True

window_name = "Debug View"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    display_img = orig_img.copy()
    if show_gt:
        for gt in gt_boxes:
            draw_box_with_label(display_img, gt[1:5], f"GT: {names[gt[0]]}", gt_colors[gt[0]], True)
    for pred in pred_boxes:
        cls = int(pred[5])
        if (cls == 0 and not show_pred_fish) or (cls == 1 and not show_pred_partial): continue
        draw_box_with_label(display_img, pred[0:4], f"{names[cls]} ({pred[4]:.2f})", pred_colors[cls], False)

    cv2.imshow(window_name, display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        show_gt = not show_gt
    elif key == ord('f'):
        show_pred_fish = not show_pred_fish
    elif key == ord('p'):
        show_pred_partial = not show_pred_partial
    elif key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()