#!/usr/bin/env python3
"""
detection_analysis.py

Run YOLO inference on a validation split and produce a detailed per-class
breakdown of TP / FP / FN, distinguishing wrong-class FPs from background FPs.

The detection pipeline mirrors extract_crops.py: a single per-class NMS runs
inside Ultralytics at INFERENCE_CONF / NMS_IOU, and the per-class confidence
thresholds (CONF_FISH / CONF_PARTIAL_FISH) are applied afterwards (post-NMS).

Matching is CLASS-AGNOSTIC and greedy one-to-one (highest confidence first, each
GT box matched at most once, any class) at IOU_THRESHOLD — the same convention as
extract_crops.py. Two views are reported: (a) per-PREDICTED-class TP/FP/FN
(detection metrics), and (b) the CROP DISTRIBUTION by matched-GT class
(fish / partial / background) — the labels the second-stage model (Model B)
actually receives. The iou_xywhn and load_gt helpers are reused from
extract_crops.py.
"""

import os

# Workaround for the Windows "libiomp5md.dll already initialized" (OMP Error #15)
# that can abort the process when multiple OpenMP runtimes get linked (torch +
# numpy/ultralytics). Must be set before torch/ultralytics import OpenMP.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from extract_crops import iou_xywhn, load_gt

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH        = "model.pt"
DATA_DIR          = "data/valid"   # contains images/ and labels/ subdirectories
CONF_FISH         = 0.005
CONF_PARTIAL_FISH = 0.005
INFERENCE_CONF    = 0.001            # low conf for inference, per-class thresholds applied post-NMS
NMS_IOU           = 0.5
IOU_THRESHOLD     = 0.5

CLASS_MAP       = {0: "fish", 1: "partial_fish"}
CONF_THRESHOLDS = {0: CONF_FISH, 1: CONF_PARTIAL_FISH}

CSV_PATH        = "detection_analysis.csv"
IMG_EXTS        = (".jpg", ".jpeg", ".png")
# ───────────────────────────────────────────────────────────────────────────────


def match_detections(
    detections: list[tuple], gt_boxes: list[tuple]
) -> tuple[list[int], set[int]]:
    """Class-agnostic greedy one-to-one matching, mirroring extract_crops.py.

    detections: list of (cls_id, conf, xc, yc, w, h)
    gt_boxes:   list of (cls_id, xc, yc, w, h)

    Each detection (highest confidence first) claims the best-IoU ground-truth
    box of ANY class among those still unmatched, accepted only if IoU >=
    IOU_THRESHOLD. This is the same matching the crop-extraction pipeline uses,
    so the resulting class distribution equals the crops fed to Model B.

    Returns (matched_gt_per_det, matched_gt_indices) where matched_gt_per_det[i]
    is the GT index matched to detection i (or -1 if it matched nothing).
    """
    matched_gt_per_det = [-1] * len(detections)
    matched_gt: set[int] = set()
    order = sorted(range(len(detections)), key=lambda i: detections[i][1], reverse=True)
    for det_idx in order:
        _, _, xc, yc, w, h = detections[det_idx]
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, (_, gxc, gyc, gw, gh) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            score = iou_xywhn((xc, yc, w, h), (gxc, gyc, gw, gh))
            if score > best_iou:
                best_iou, best_gt_idx = score, gt_idx
        if best_iou >= IOU_THRESHOLD and best_gt_idx != -1:
            matched_gt_per_det[det_idx] = best_gt_idx
            matched_gt.add(best_gt_idx)
    return matched_gt_per_det, matched_gt


def analyze_image(detections: list[tuple], gt_boxes: list[tuple]) -> dict:
    """Categorise every detection and unmatched GT, class-agnostically.

    Matching is class-agnostic (a detection may claim a GT box of any class),
    mirroring extract_crops.py. Two complementary views are produced:

    * Per PREDICTED class (detection metrics): each detection is a TP if it
      claimed a GT of the SAME class, a wrong-class FP if it claimed a GT of a
      different class, or a background FP if it claimed nothing.
    * CROP DISTRIBUTION (what Model B sees): each detection is labelled by the
      class of the GT it claimed (fish / partial), or background if it claimed
      nothing — identical to extract_crops.py's assign_labels_greedy.

    Returns a dict of integer counts for this single image.
    """
    matched_gt_per_det, matched_gt = match_detections(detections, gt_boxes)

    counts = {
        "tp_fish": 0, "fp_fish_wrongclass": 0, "fp_fish_bg": 0, "fn_fish": 0,
        "tp_partial": 0, "fp_partial_wrongclass": 0, "fp_partial_bg": 0, "fn_partial": 0,
        # Crop distribution by matched-GT class (= the labels Model B receives)
        "crop_fish": 0, "crop_partial": 0, "crop_background": 0,
    }

    for det_idx, (det_cls, _, _, _, _, _) in enumerate(detections):
        gt_idx = matched_gt_per_det[det_idx]
        suffix = "fish" if det_cls == 0 else "partial"
        if gt_idx == -1:
            counts[f"fp_{suffix}_bg"] += 1
            counts["crop_background"] += 1
            continue
        gt_cls = gt_boxes[gt_idx][0]
        # Detection-metric view (by predicted class)
        if gt_cls == det_cls:
            counts[f"tp_{suffix}"] += 1
        else:
            counts[f"fp_{suffix}_wrongclass"] += 1
        # Crop-distribution view (by the class of the GT actually claimed)
        if gt_cls == 0:
            counts["crop_fish"] += 1
        elif gt_cls == 1:
            counts["crop_partial"] += 1

    for gt_idx, (gt_cls, *_rest) in enumerate(gt_boxes):
        if gt_idx in matched_gt:
            continue
        if gt_cls == 0:
            counts["fn_fish"] += 1
        elif gt_cls == 1:
            counts["fn_partial"] += 1

    # Derived FP totals
    counts["fp_fish_total"] = counts["fp_fish_wrongclass"] + counts["fp_fish_bg"]
    counts["fp_partial_total"] = counts["fp_partial_wrongclass"] + counts["fp_partial_bg"]
    return counts


def run_inference(model: YOLO, img_path: Path, gt_label_dir: Path) -> tuple[list[tuple], list[tuple]]:
    """Run inference + per-class thresholding for one image; return (detections, gt_boxes)."""
    results = model(str(img_path), verbose=False,
                    conf=INFERENCE_CONF, iou=NMS_IOU, max_det=1000)
    gt_boxes = load_gt(gt_label_dir / (img_path.stem + ".txt"))

    detections: list[tuple] = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if conf >= CONF_THRESHOLDS.get(cls_id, 1.0):
                xc, yc, w, h = box.xywhn[0].tolist()
                detections.append((cls_id, conf, xc, yc, w, h))
    return detections, gt_boxes


def yolo_val_crosscheck(model: YOLO, data_yaml: Path, split: str, val_conf: float) -> dict | None:
    """Run model.val() and extract per-class TP/FP/FN from its confusion matrix.

    The confusion matrix is laid out matrix[predicted_class, gt_class] with the
    last index (nc) representing background. For class i:
        TP = M[i, i]
        FP = sum(M[i, :]) - M[i, i]   (predicted i, GT was other class or background)
        FN = sum(M[:, i]) - M[i, i]   (GT i, predicted other class or background)

    Returns {0: (tp, fp, fn), 1: (tp, fp, fn)} or None on failure.
    """
    metrics = model.val(data=str(data_yaml), split=split,
                        conf=val_conf, iou=NMS_IOU, verbose=False)
    cm = metrics.confusion_matrix.matrix  # shape (nc+1, nc+1)
    out: dict[int, tuple[int, int, int]] = {}
    for cls in (0, 1):
        tp = float(cm[cls, cls])
        fp = float(cm[cls, :].sum()) - tp
        fn = float(cm[:, cls].sum()) - tp
        out[cls] = (int(round(tp)), int(round(fp)), int(round(fn)))
    return out


def main() -> None:
    base_dir = Path(__file__).parent
    data_dir = base_dir / DATA_DIR
    img_dir  = data_dir / "images"
    lbl_dir  = data_dir / "labels"

    model_path = base_dir / MODEL_PATH
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"No images found in {img_dir}")
        return

    print(f"Model : {model_path.name}")
    print(f"Data  : {img_dir}  ({len(images)} images)")
    print(f"Conf  : fish={CONF_FISH}  partial_fish={CONF_PARTIAL_FISH}  "
          f"(inference={INFERENCE_CONF}, NMS IoU={NMS_IOU}, match IoU={IOU_THRESHOLD})")
    print(f"Running inference on {len(images)} image(s) …\n")

    model = YOLO(str(model_path))

    rows: list[dict] = []
    for img_path in images:
        detections, gt_boxes = run_inference(model, img_path, lbl_dir)
        counts = analyze_image(detections, gt_boxes)
        rows.append({
            "image_name":            img_path.name,
            "tp_fish":               counts["tp_fish"],
            "fp_fish_total":         counts["fp_fish_total"],
            "fp_fish_wrongclass":    counts["fp_fish_wrongclass"],
            "fp_fish_bg":            counts["fp_fish_bg"],
            "fn_fish":               counts["fn_fish"],
            "tp_partial":            counts["tp_partial"],
            "fp_partial_total":      counts["fp_partial_total"],
            "fp_partial_wrongclass": counts["fp_partial_wrongclass"],
            "fp_partial_bg":         counts["fp_partial_bg"],
            "fn_partial":            counts["fn_partial"],
            "crop_fish":             counts["crop_fish"],
            "crop_partial":          counts["crop_partial"],
            "crop_background":       counts["crop_background"],
        })

    df = pd.DataFrame(rows)
    sum_row = df.sum(numeric_only=True).to_frame().T
    sum_row.insert(0, "image_name", "TOTAL")
    df_with_sum = pd.concat([df, sum_row], ignore_index=True)
    csv_path = base_dir / CSV_PATH
    df_with_sum.to_csv(csv_path, index=False)
    print(f"Per-image results → {csv_path}\n")

    # ── Aggregate totals ────────────────────────────────────────────────────────
    agg = {
        "fish": {
            "tp": int(df["tp_fish"].sum()),
            "fp_total": int(df["fp_fish_total"].sum()),
            "fp_wrong": int(df["fp_fish_wrongclass"].sum()),
            "fp_bg": int(df["fp_fish_bg"].sum()),
            "fn": int(df["fn_fish"].sum()),
        },
        "partial_fish": {
            "tp": int(df["tp_partial"].sum()),
            "fp_total": int(df["fp_partial_total"].sum()),
            "fp_wrong": int(df["fp_partial_wrongclass"].sum()),
            "fp_bg": int(df["fp_partial_bg"].sum()),
            "fn": int(df["fn_partial"].sum()),
        },
    }

    # ── Summary table ───────────────────────────────────────────────────────────
    print("=" * 100)
    print("SUMMARY (custom analysis)")
    print("=" * 100)
    header = (f"{'Class':<13} {'TP':>5} {'FP(total)':>10} {'FP(wrong class)':>16} "
              f"{'FP(background)':>15} {'FN':>5} {'Precision':>10} {'Recall':>8}")
    print(header)
    for cls_name, c in agg.items():
        tp, fp_total, fn = c["tp"], c["fp_total"], c["fn"]
        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"{cls_name:<13} {tp:>5} {fp_total:>10} {c['fp_wrong']:>16} "
              f"{c['fp_bg']:>15} {fn:>5} {precision:>10.4f} {recall:>8.4f}")
    print()

    # ── Crop distribution (what Model B receives) ───────────────────────────────
    crop_fish = int(df["crop_fish"].sum())
    crop_partial = int(df["crop_partial"].sum())
    crop_bg = int(df["crop_background"].sum())
    crop_total = crop_fish + crop_partial + crop_bg
    print("=" * 100)
    print("CROP DISTRIBUTION  (by matched-GT class — the labels Model B receives)")
    print("=" * 100)
    print(f"  fish        : {crop_fish}")
    print(f"  partial_fish: {crop_partial}")
    print(f"  background  : {crop_bg}")
    print(f"  TOTAL       : {crop_total}")
    print()

    # ── Sanity checks ───────────────────────────────────────────────────────────
    print("=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)
    det_totals = {
        "fish": int((df["tp_fish"] + df["fp_fish_total"]).sum()),
        "partial_fish": int((df["tp_partial"] + df["fp_partial_total"]).sum()),
    }
    all_ok = True
    for cls_name, c in agg.items():
        tp, fp_total, fp_wrong, fp_bg = c["tp"], c["fp_total"], c["fp_wrong"], c["fp_bg"]
        n_det = det_totals[cls_name]

        if tp + fp_total != n_det:
            print(f"  WARNING [{cls_name}]: TP + FP(total) = {tp + fp_total} "
                  f"!= total detections above threshold ({n_det})")
            all_ok = False
        else:
            print(f"  OK [{cls_name}]: TP + FP(total) = total detections = {n_det}")

        if fp_wrong + fp_bg != fp_total:
            print(f"  WARNING [{cls_name}]: FP(wrong class) + FP(background) = "
                  f"{fp_wrong + fp_bg} != FP(total) ({fp_total})")
            all_ok = False
        else:
            print(f"  OK [{cls_name}]: FP(wrong class) + FP(background) = FP(total) = {fp_total}")

    total_det = det_totals["fish"] + det_totals["partial_fish"]
    if crop_total != total_det:
        print(f"  WARNING [crops]: crop_fish+crop_partial+crop_background = {crop_total} "
              f"!= total detections ({total_det})")
        all_ok = False
    else:
        print(f"  OK [crops]: fish+partial+background crops = total detections = {total_det}")
    if all_ok:
        print("  All sanity checks passed.")
    print()

    # ── YOLO built-in validation cross-check ────────────────────────────────────
    print("=" * 100)
    print("YOLO model.val() CROSS-CHECK")
    print("=" * 100)
    val_conf = min(CONF_FISH, CONF_PARTIAL_FISH)
    data_yaml = data_dir.parent / "data.yaml"
    print(f"Note: exact agreement is NOT expected — the two methods account differently:")
    print(f"      (1) model.val() applies a SINGLE conf threshold to all classes, so it is run")
    print(f"          at conf={val_conf} (= min of the per-class thresholds); the custom analysis")
    print(f"          uses per-class thresholds (fish={CONF_FISH}, partial={CONF_PARTIAL_FISH}).")
    print(f"      (2) The custom analysis matches CLASS-AGNOSTICALLY (like extract_crops.py):")
    print(f"          TP/FP/FN are reported by PREDICTED class, so a fish GT claimed by a")
    print(f"          partial-predicted box is a wrong-class FP, not a fish TP. The CROP")
    print(f"          DISTRIBUTION above (by matched-GT class) is what Model B actually sees.")
    print(f"      (3) val's confusion-matrix matching IoU may differ slightly from "
          f"IOU_THRESHOLD={IOU_THRESHOLD}.\n")

    val_counts = None
    if not data_yaml.exists():
        print(f"  data.yaml not found at {data_yaml}; skipping val() cross-check.")
    else:
        try:
            val_counts = yolo_val_crosscheck(model, data_yaml, "val", val_conf)
        except Exception as exc:  # noqa: BLE001 — cross-check must not break the run
            print(f"  model.val() failed: {exc}")

    if val_counts is not None:
        print(f"{'Class':<13} {'Source':<10} {'TP':>6} {'FP':>6} {'FN':>6}")
        for cls_id, cls_name in CLASS_MAP.items():
            c = agg[cls_name]
            v_tp, v_fp, v_fn = val_counts[cls_id]
            print(f"{cls_name:<13} {'custom':<10} {c['tp']:>6} {c['fp_total']:>6} {c['fn']:>6}")
            print(f"{'':<13} {'yolo.val':<10} {v_tp:>6} {v_fp:>6} {v_fn:>6}")
    print("\nDone.")


if __name__ == "__main__":
    main()
