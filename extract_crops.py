#!/usr/bin/env python3
"""
extract_crops.py

Run model A (the YOLO detector) on each dataset split, extract padded crops
from the original 16:9 images, and write labels.csv — the training set for
model B (the ConvNeXt crop classifier: fish / partial_fish / background).

Labeling philosophy (image-classifier variant)
-----------------------------------------------
Model B sees one CROP at inference and answers "what is in this crop?". It does
NOT know that two overlapping crops belong to the same physical object, and it
does NOT deduplicate — duplicate suppression is the job of the FINAL NMS that
runs AFTER model B.

Therefore each detection is labeled INDEPENDENTLY by the class of the GT object
it overlaps (best IoU >= IOU_THRESHOLD), with NO one-to-one matching:
  • a crop on a real fish            → its GT class (fish / partial_fish),
    even if it is a duplicate detection of that same fish.
  • a crop on nothing (pipe, bg)     → "background".

This is deliberately different from YOLO's official one-to-one eval matching:
that scheme would force a duplicate crop of a real fish into "background",
creating near-identical crops with opposite labels — contradictory supervision
for an image classifier. See assign_labels_by_overlap().

Representing real inference
---------------------------
The crops must reflect model A's REAL output. Two requirements:
  1. Run model A here with the SAME NMS settings as production inference
     (conf / iou / agnostic). Per the agreed architecture, model A is
     CLASS-AWARE (AGNOSTIC_NMS = False): a fish box and a partial_fish box on
     the same object both survive. Keep this identical to deployment.
  2. Feed only images model A did NOT train on (handled by the input folders
     you point this script at), or the false-positive rate will be unrealistic.

Set SAMPLE_MODE = True to preview a few images with overlay before the full run
(press any key to advance, 'q' to quit the preview early).
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
PADDING           = 0.10   # expand each box edge by this fraction of box size
CONF_FISH         = 0.02   # confidence threshold for class 0 (fish)
CONF_PARTIAL_FISH = 0.07    # confidence threshold for class 1 (partial_fish)
SAMPLE_MODE       = False  # process only a few images + show overlay
SAMPLE_SIZE       = 5      # first N images from the sorted dataset

# Detection pipeline — must mirror model A's PRODUCTION inference, not val().
# Inference runs NMS once at INFERENCE_CONF / NMS_IOU / AGNOSTIC_NMS. The per-class
# CONF_THRESHOLDS above are applied AFTERWARDS (post-NMS), so changing them only
# selects which surviving detections become crops — it does NOT change the NMS
# outcome. Keep INFERENCE_CONF low so NMS sees the full candidate set first.
INFERENCE_CONF    = 0.001  # conf passed to model() — keep low, threshold later
NMS_IOU           = 0.5    # iou passed to model() — match deployment
AGNOSTIC_NMS      = False  # class-AWARE NMS: keep cross-class duplicates (fish +
                           # partial_fish on the same object both survive). The
                           # FINAL NMS after model B handles dedup. Must match
                           # production; flip to True ONLY if deployment uses it.

# Leave empty to process the full split (or SAMPLE_SIZE images when SAMPLE_MODE=True).
# Populate with image stems (no extension) to process specific images only, e.g.:
#   CUSTOM_IMAGES = ["frame_00042", "frame_01337"]
CUSTOM_IMAGES: list[str] = []

IOU_THRESHOLD     = 0.5    # IoU above which a crop is considered to contain a GT object
DEBUG_MODE        = False  # print per-detection matching info when True

CLASS_MAP       = {0: "fish", 1: "partial_fish"}
CONF_THRESHOLDS = {0: CONF_FISH, 1: CONF_PARTIAL_FISH}

DATASETS = [
    ("data", "model.pt"),
    #("data12", "model12.pt"),
    #("data15", "model15.pt"),
    #("data25", "model25.pt"),
]
SPLIT = "valid"
# ───────────────────────────────────────────────────────────────────────────────


# def apply_filters(image: np.ndarray) -> np.ndarray:
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl, a, b))
#     clahed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     clahed_then_blurred = cv2.GaussianBlur(clahed, (55, 55), 0)
#     sharpened = cv2.addWeighted(clahed, 1.8, clahed_then_blurred, -0.8, 0)
#     return sharpened

def apply_filters(image: np.ndarray) -> np.ndarray:
    return image


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


def assign_labels_by_overlap(
    detections: list[tuple], gt_boxes: list[tuple]
) -> list[str]:
    """Label each detection INDEPENDENTLY by the class of its best-overlapping GT.

    No one-to-one constraint: a GT object may "claim" several detections, so a
    duplicate crop of a real fish still gets the fish label (as an image, it does
    contain a fish). Only detections that overlap no GT above IOU_THRESHOLD become
    "background". This is the correct supervision for an image classifier whose
    deduplication happens later in the final NMS.

    Returns labels in the same order as the input detections.
    """
    labels: list[str] = []
    for (_, _, xc, yc, w, h) in detections:
        best_iou, best_cls = 0.0, None
        for (gcls, gxc, gyc, gw, gh) in gt_boxes:
            score = iou_xywhn((xc, yc, w, h), (gxc, gyc, gw, gh))
            if score > best_iou:
                best_iou, best_cls = score, gcls
        if best_iou >= IOU_THRESHOLD and best_cls is not None:
            labels.append(CLASS_MAP.get(best_cls, str(best_cls)))
        else:
            labels.append("background")
    return labels


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
    else:
        subset = images
    print(f"  Processing {len(subset)} image(s) …")

    orig_lookup = build_orig_lookup(original_img_dir)

    model = YOLO(str(model_path))
    rows: list[dict] = []
    crop_counter: dict[str, int] = {}

    for img_path in subset:
        stem = img_path.stem

        # NMS runs once inside Ultralytics with the deployment settings.
        # AGNOSTIC_NMS=False → class-aware: cross-class duplicates survive.
        results  = model(str(img_path), verbose=False,
                         conf=INFERENCE_CONF, iou=NMS_IOU,
                         agnostic_nms=AGNOSTIC_NMS, max_det=1000)
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

        # NMS already happened inside model(). Keep only survivors whose confidence
        # clears the per-class threshold.
        detections: list[tuple] = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if conf >= CONF_THRESHOLDS.get(cls_id, 1.0):
                    xc, yc, w, h = box.xywhn[0].tolist()
                    detections.append((cls_id, conf, xc, yc, w, h))

        labels = assign_labels_by_overlap(detections, gt_boxes)

        if DEBUG_MODE:
            print(f"\n  [DEBUG] {img_path.name}")
            print(f"    detections after NMS: {len(detections)}, GT boxes: {len(gt_boxes)}")
            for i, (cls_id, conf, xc, yc, w, h) in enumerate(detections):
                best_iou = max(
                    (iou_xywhn((xc, yc, w, h), (gxc, gyc, gw, gh))
                     for _, gxc, gyc, gw, gh in gt_boxes),
                    default=0.0,
                )
                print(f"    det[{i}] conf={conf:.4f} pred={CLASS_MAP.get(cls_id, str(cls_id))} "
                      f"label={labels[i]} best_iou={best_iou:.4f}")

        vis = orig.copy() if SAMPLE_MODE else None

        idx = crop_counter.get(stem, 0)
        for (cls_id, conf, xc, yc, w, h), label in zip(detections, labels):
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

            # Tabular features for model B variant B (image + metadata) and model C.
            box_area     = w * h                              # normalized area
            long_side    = max(w, h)
            short_side   = min(w, h)
            aspect_ratio = (long_side / short_side) if short_side > 0 else 0.0  # always >= 1

            rows.append({
                "image_name":    img_path.name,
                "crop_filename": crop_fname,
                "class_id":      cls_id,                       # model A's predicted class
                "class_name":    CLASS_MAP.get(cls_id, str(cls_id)),
                "confidence":    round(conf, 4),
                "x_center":      round(xc, 6),
                "y_center":      round(yc, 6),
                "width":         round(w, 6),
                "height":        round(h, 6),
                "box_area":      round(box_area, 6),
                "aspect_ratio":  round(aspect_ratio, 4),
                "label":         label,                        # GT-based target for model B
            })

            if SAMPLE_MODE and vis is not None:
                color = (0, 220, 0) if cls_id == 0 else (0, 140, 255)
                cv2.rectangle(vis, (x1p, y1p), (x2p, y2p), color, 5)
                tag = f"{CLASS_MAP.get(cls_id, str(cls_id))} {conf:.2f} -> {label}"
                cv2.putText(vis, tag, (x1p, max(y1p - 12, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

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
        n_bg = sum(1 for r in rows if r["label"] == "background")
        print(f"  {len(rows)} crop(s) saved → {out_dir}   "
              f"({len(rows) - n_bg} object / {n_bg} background)")
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