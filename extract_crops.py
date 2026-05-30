#!/usr/bin/env python3
"""
extract_crops.py

Run YOLO inference on the val split of each dataset, extract padded crops
from the original 16:9 images, apply CLAHE+sharpening, and write labels.csv.

Set SAMPLE_MODE = True to preview 20 random images with overlay before the
full run (press any key to advance, 'q' to quit the preview early).
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
PADDING           = 0.1   # expand each box edge by this fraction of box size
CONF_FISH         = 0.01   # confidence threshold for class 0 (fish)
CONF_PARTIAL_FISH = 0.09   # confidence threshold for class 1 (partial_fish)
SAMPLE_MODE       = True   # process only SAMPLE_SIZE images + show overlay
SAMPLE_SIZE       = 5      # first N images from the sorted dataset

# Leave empty to process the full split (or SAMPLE_SIZE images when SAMPLE_MODE=True).
# Populate with image stems (no extension) to process specific images only, e.g.:
#   CUSTOM_IMAGES = ["frame_00042", "frame_01337"]
CUSTOM_IMAGES: list[str] = []

IOU_THRESHOLD   = 0.5

CLASS_MAP       = {0: "fish", 1: "partial_fish"}
CONF_THRESHOLDS = {0: CONF_FISH, 1: CONF_PARTIAL_FISH}

DATASETS = [
    ("data12", "model12.pt"),
    # ("data15", "model15.pt"),
    # ("data25", "model25.pt"),
]
SPLIT = "valid"
# ───────────────────────────────────────────────────────────────────────────────


def apply_filters(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    clahed_then_blurred = cv2.GaussianBlur(clahed, (55, 55), 0)
    sharpened = cv2.addWeighted(clahed, 1.8, clahed_then_blurred, -0.8, 0)
    return sharpened

# def apply_filters(image: np.ndarray) -> np.ndarray:
#     return image


def iou_xywhn(b1: tuple, b2: tuple) -> float:
    """IoU for two boxes in normalized (xc, yc, w, h) format."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1 = max(x1 - w1 / 2, x2 - w2 / 2)
    iy1 = max(y1 - h1 / 2, y2 - h2 / 2)
    ix2 = min(x1 + w1 / 2, x2 + w2 / 2)
    iy2 = min(y1 + h1 / 2, y2 + h2 / 2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path: Path) -> list[tuple]:
    """Load YOLO GT file → list of (class_id, xc, yc, w, h). Empty if file missing."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return boxes


def assign_label(det: tuple, gt_boxes: list[tuple]) -> str:
    """Return GT class_name for the highest-IoU match >= IOU_THRESHOLD, else 'background'."""
    best_iou, best_cls = 0.0, None
    for cls_id, gxc, gyc, gw, gh in gt_boxes:
        score = iou_xywhn(det, (gxc, gyc, gw, gh))
        if score > best_iou:
            best_iou, best_cls = score, cls_id
    if best_iou >= IOU_THRESHOLD and best_cls is not None:
        return CLASS_MAP.get(best_cls, str(best_cls))
    return "background"

def expand_box(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int, padding: float,
) -> tuple[int, int, int, int]:
    """Expand box by `padding` fraction of its own dimensions, clamped to image."""
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0,     int(x1 - bw * padding))
    y1 = max(0,     int(y1 - bh * padding))
    x2 = min(img_w, int(x2 + bw * padding))
    y2 = min(img_h, int(y2 + bh * padding))
    return x1, y1, x2, y2


def original_stem(stem: str) -> str:
    """Strip Roboflow suffix: 'name_jpg.rf.abc123' → 'name'."""
    return re.sub(r'[_.](?:jpg|jpeg|png)\.rf\.[a-f0-9]+$', '', stem, flags=re.IGNORECASE)


def build_orig_lookup(orig_dir: Path) -> dict[str, Path]:
    """Map base stem (Roboflow hash stripped) → Path for every file in orig_dir."""
    lookup: dict[str, Path] = {}
    if not orig_dir.exists():
        return lookup
    for p in orig_dir.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            lookup[original_stem(p.stem)] = p
    return lookup


def process_dataset(dataset_name: str, model_path: Path, base_dir: Path) -> None:
    processed_img_dir = base_dir / dataset_name / SPLIT / "images"
    original_img_dir  = base_dir / f"{dataset_name}_original" / SPLIT / "images"
    gt_label_dir      = base_dir / dataset_name / SPLIT / "labels"
    out_dir           = base_dir / "crops" / dataset_name / SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(processed_img_dir.glob("*.jpg")) +
        list(processed_img_dir.glob("*.jpeg")) +
        list(processed_img_dir.glob("*.png"))
    )
    if not images:
        print(f"  No images found in {processed_img_dir}")
        return

    if CUSTOM_IMAGES:
        stems  = set(CUSTOM_IMAGES)
        subset = [p for p in images if p.stem in stems]
        missing = stems - {p.stem for p in subset}
        if missing:
            print(f"  Warning: images not found in {processed_img_dir}: {missing}")
    elif SAMPLE_MODE:
        subset = images[:SAMPLE_SIZE]
        subset = images[44:49]
    else:
        subset = images
    print(f"  Processing {len(subset)} image(s) …")

    orig_lookup = build_orig_lookup(original_img_dir)

    model = YOLO(str(model_path))
    rows: list[dict] = []
    crop_counter: dict[str, int] = {}

    for img_path in subset:
        stem = img_path.stem

        results  = model(str(img_path), verbose=False)
        gt_boxes = load_gt(gt_label_dir / (stem + ".txt"))

        orig_path = orig_lookup.get(original_stem(stem))
        if orig_path is None:
            print(f"  Original not found for {img_path.name}, skipping.")
            continue
        orig = cv2.imread(str(orig_path))
        if orig is None:
            print(f"  Could not read {orig_path}, skipping.")
            continue
        oh, ow = orig.shape[:2]

        # Filter detections by per-class confidence threshold
        detections: list[tuple] = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if conf >= CONF_THRESHOLDS.get(cls_id, 1.0):
                    xc, yc, w, h = box.xywhn[0].tolist()
                    detections.append((cls_id, conf, xc, yc, w, h))

        vis = orig.copy() if SAMPLE_MODE else None

        idx = crop_counter.get(stem, 0)
        for cls_id, conf, xc, yc, w, h in detections:
            # Relative coords are identical in original image (stretched 1:1 → 16:9)
            x1_abs = (xc - w / 2) * ow
            y1_abs = (yc - h / 2) * oh
            x2_abs = (xc + w / 2) * ow
            y2_abs = (yc + h / 2) * oh

            x1p, y1p, x2p, y2p = expand_box(x1_abs, y1_abs, x2_abs, y2_abs, ow, oh, PADDING)

            crop = orig[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            crop = apply_filters(crop)
            crop_fname = f"{stem}_crop{idx:03d}.jpg"
            cv2.imwrite(str(out_dir / crop_fname), crop)

            rows.append({
                "image_name":    img_path.name,
                "crop_filename": crop_fname,
                "class_id":      cls_id,
                "class_name":    CLASS_MAP.get(cls_id, str(cls_id)),
                "confidence":    round(conf, 4),
                "x_center":      round(xc, 6),
                "y_center":      round(yc, 6),
                "width":         round(w, 6),
                "height":        round(h, 6),
                "label":         assign_label((xc, yc, w, h), gt_boxes),
            })

            if SAMPLE_MODE and vis is not None:
                color = (0, 220, 0) if cls_id == 0 else (0, 140, 255)
                cv2.rectangle(vis, (x1p, y1p), (x2p, y2p), color, 5)
                tag = f"{CLASS_MAP.get(cls_id, str(cls_id))} {conf:.2f}"
                cv2.putText(vis, tag, (x1p, max(y1p - 12, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)

            idx += 1
        crop_counter[stem] = idx

        if SAMPLE_MODE and vis is not None:
            scale = min(1280 / ow, 800 / oh, 1.0)
            disp  = cv2.resize(vis, (int(ow * scale), int(oh * scale)))
            title = f"[{dataset_name}] {img_path.name}  (any key=next  q=quit)"
            cv2.imshow(title, disp)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                print("  Preview quit by user.")
                break

    # Write or append CSV (deduplicate by crop_filename, keep latest)
    csv_path = out_dir / "labels.csv"
    if rows:
        new_df = pd.DataFrame(rows)
        if csv_path.exists():
            old_df  = pd.read_csv(csv_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["crop_filename"], keep="last")
        else:
            combined = new_df
        combined.to_csv(csv_path, index=False)
        print(f"  {len(rows)} crop(s) saved → {out_dir}")
        print(f"  CSV → {csv_path}")
    else:
        print("  No detections above threshold.")


def main() -> None:
    base_dir = Path(__file__).parent
    for dataset_name, model_file in DATASETS:
        model_path = base_dir / model_file
        if not model_path.exists():
            print(f"\nModel not found: {model_path}  (skipping {dataset_name})")
            continue
        print(f"\n{'─' * 60}")
        print(f"{dataset_name}  ←  {model_file}")
        process_dataset(dataset_name, model_path, base_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
