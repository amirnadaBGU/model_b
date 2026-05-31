#!/usr/bin/env python3
"""
export_to_roboflow.py

Reads labels.csv from each dataset/split and copies crops into a
class-folder structure ready for Roboflow classification upload.

Output:
    roboflow_export/
    ├── fish/
    ├── partial_fish/
    └── background/
"""

from pathlib import Path
import shutil
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
CROPS_BASE    = Path("crops")
EXPORT_DIR    = Path("roboflow_export")
DATASETS      = ["data12", "data15", "data25"]
SPLITS        = ["valid"]          # add "train" etc. if needed
ONLY_APPROVED = False               # if True, skip rows where review_status != "approved"
# ───────────────────────────────────────────────────────────────────────────────

CLASS_OPTIONS = ["fish", "partial_fish", "background"]

def main() -> None:
    for cls in CLASS_OPTIONS:
        (EXPORT_DIR / cls).mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0

    for dataset in DATASETS:
        for split in SPLITS:
            csv_path = CROPS_BASE / dataset / split / "labels.csv"
            img_dir  = CROPS_BASE / dataset / split

            if not csv_path.exists():
                print(f"  Skipping {dataset}/{split} — labels.csv not found")
                continue

            df = pd.read_csv(csv_path, dtype={"label": str, "review_status": str})

            if ONLY_APPROVED:
                df = df[df["review_status"] == "approved"]

            for _, row in df.iterrows():
                label = str(row.get("label", "")).strip()
                if label not in CLASS_OPTIONS:
                    skipped += 1
                    continue

                src = img_dir / row["crop_filename"]
                if not src.exists():
                    print(f"  Missing file: {src}")
                    skipped += 1
                    continue

                # Prefix with dataset name to avoid filename collisions
                dst_name = f"{dataset}_{row['crop_filename']}"
                dst = EXPORT_DIR / label / dst_name
                shutil.copy2(src, dst)
                copied += 1

    print(f"\nDone. {copied} crops exported, {skipped} skipped.")
    for cls in CLASS_OPTIONS:
        n = len(list((EXPORT_DIR / cls).glob("*.jpg")))
        print(f"  {cls}: {n}")

if __name__ == "__main__":
    main()