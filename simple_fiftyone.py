import os
import fiftyone as fo
import fiftyone.utils.ultralytics as fou
from ultralytics import YOLO

# 1. הגדרות בסיסיות ודרכים לקבצים
fo.config.database_validation = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CONF_THRESHOLD = 0.436

model = YOLO("version6.pt")
classes = model.names

images_dir = os.path.abspath("datasets/data/valid/images")
labels_dir = os.path.abspath("datasets/data/valid/labels")

# 2. יצירת הדאטהסט ב-FiftyOne
dataset_name = "fish_predict_eval"
if fo.dataset_exists(dataset_name):
    fo.load_dataset(dataset_name).delete()

print("Creating dataset...")
dataset = fo.Dataset(name=dataset_name)
dataset.add_dir(dataset_dir=images_dir, dataset_type=fo.types.ImageDirectory)

# 3. טעינת תגיות האמת (Ground Truth) מקובצי ה-TXT
print("Loading Ground Truth labels...")
for sample in dataset:
    img_name = os.path.splitext(os.path.basename(sample.filepath))[0]
    label_path = os.path.join(labels_dir, img_name + ".txt")

    if os.path.exists(label_path):
        detections = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, xc, yc, w, h = map(float, parts)
                    detections.append(
                        fo.Detection(
                            label=classes[int(c)],
                            bounding_box=[xc - w / 2, yc - h / 2, w, h]
                        )
                    )
        sample["ground_truth"] = fo.Detections(detections=detections)
        sample.save()

# 4. הרצת ה-Predict עם ה-Confidence הנבחר
print(f"Running model predict (Conf={CONF_THRESHOLD})...")
for sample in dataset:
    result = model.predict(
        sample.filepath,
        agnostic_nms=False,
        conf=CONF_THRESHOLD,
        iou=0.5,
        imgsz=640,
        verbose=False,
        device="cuda:0"
    )[0]

    # המרה אוטומטית של תוצאות YOLO לפורמט של FiftyOne
    detections = fou.to_detections(result)
    if detections is None:
        detections = fo.Detections(detections=[])

    sample["predictions"] = detections
    sample.save()

# 5. חישוב והדפסת התוצאות ומטריצת הבלבול
print("\n=== EVALUATION RESULTS ===")
eval_results = dataset.evaluate_detections(
    pred_field="predictions",
    gt_field="ground_truth",
    eval_key="eval"
)

# הדפסת דוח דיוק (Precision, Recall, F1)
print("\nClassification Report:")
print(eval_results.report())

# הדפסת מטריצת הבלבול ישירות לטרמינל
print("\nConfusion Matrix:")
print(eval_results.confusion_matrix())

# 6. הפעלת הממשק הגרפי (ה-App)
print("\nLaunching FiftyOne App...")
session = fo.launch_app(dataset)
session.wait()