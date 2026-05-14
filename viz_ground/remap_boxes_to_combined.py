#!/usr/bin/env python
"""Remap dscope/clinical grounding boxes into combined-image coordinates.

For each combined row in *_predictions_reason_viz.csv, this script:
  1. Opens the photo and dscope images to compute the split ratio
     (photo_width / combined_width after height-matching resize).
  2. Remaps each box in viz_grounding_clinical into the LEFT half
     of the combined image coordinate space.
  3. Remaps each box in viz_grounding_dscope into the RIGHT half.
  4. Writes two new columns:
       viz_grounding_clinical_remapped  — boxes in combined-image coords
       viz_grounding_dscope_remapped    — boxes in combined-image coords

Non-combined rows get empty strings for these columns.

Usage:
    python viz_ground/remap_boxes_to_combined.py                   # all _viz CSVs
    python viz_ground/remap_boxes_to_combined.py --csv dermato_llama
"""

import argparse
import glob
import json
import os
import sys

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")


def compute_split_ratio(photo_path: str, dscope_path: str) -> float:
    """Return photo_width / (photo_width + dscope_width) after height-matching.

    Mirrors the resize logic in data_utils/utils.py prepare_all_lesions.
    """
    photo = Image.open(photo_path)
    dscope = Image.open(dscope_path)
    h = min(photo.height, dscope.height)
    pw = int(photo.width * h / photo.height)
    dw = int(dscope.width * h / dscope.height)
    return pw / (pw + dw)


def remap_box(box: dict, split_ratio: float, side: str) -> dict:
    """Remap a normalized [0,1] box from a single image into combined coords.

    side='clinical' maps into [0, split_ratio] on x-axis.
    side='dscope'   maps into [split_ratio, 1] on x-axis.
    y/h are unchanged (same height).
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    if side == "clinical":
        return {
            "x": round(x * split_ratio, 4),
            "y": round(y, 4),
            "w": round(w * split_ratio, 4),
            "h": round(h, 4),
        }
    else:
        return {
            "x": round(split_ratio + x * (1 - split_ratio), 4),
            "y": round(y, 4),
            "w": round(w * (1 - split_ratio), 4),
            "h": round(h, 4),
        }


def remap_entries(entries_json: str, split_ratio: float, side: str) -> str:
    """Remap all boxes in a viz_grounding JSON column."""
    if not entries_json or entries_json in ("", "[]"):
        return "[]"
    try:
        entries = json.loads(entries_json)
    except (json.JSONDecodeError, TypeError):
        return "[]"

    remapped = []
    for entry in entries:
        out = dict(entry)
        if entry.get("box") and isinstance(entry["box"], dict):
            out["box"] = remap_box(entry["box"], split_ratio, side)
        remapped.append(out)
    return json.dumps(remapped, separators=(",", ":"))


def find_viz_csvs(results_dir: str, filter_model: str | None = None) -> list[tuple[str, str]]:
    """Return (model_name, path) for each *_predictions_reason_viz.csv."""
    pattern = os.path.join(results_dir, "*_predictions_reason_viz.csv")
    pairs = []
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        model_name = basename.replace("_predictions_reason_viz.csv", "")
        if filter_model and model_name != filter_model:
            continue
        pairs.append((model_name, path))
    return pairs


def process_csv(model_name: str, csv_path: str, images_dir: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({len(df)} rows)")

    needed_cols = {"viz_grounding_dscope", "viz_grounding_clinical", "image_mode"}
    if not needed_cols.issubset(df.columns):
        missing = needed_cols - set(df.columns)
        print(f"  SKIP — missing columns: {missing}")
        return

    combined = df[df["image_mode"] == "combined"]
    print(f"  Combined rows: {len(combined)}")

    if "viz_grounding_clinical_remapped" in df.columns:
        already = combined["viz_grounding_clinical_remapped"].notna() & (combined["viz_grounding_clinical_remapped"] != "")
        print(f"  Already remapped: {already.sum()}/{len(combined)}")
    else:
        df["viz_grounding_clinical_remapped"] = ""
        df["viz_grounding_dscope_remapped"] = ""

    # Cache split ratios by lesion number
    split_cache: dict[str, float] = {}
    skipped = 0
    remapped = 0

    for idx, row in combined.iterrows():
        # Skip if already done
        existing = df.at[idx, "viz_grounding_clinical_remapped"]
        if existing and existing != "" and not (isinstance(existing, float)):
            continue

        case_id = row["id"]  # e.g. "123_combined"
        num = case_id.replace("_combined", "")

        if num not in split_cache:
            photo_path = os.path.join(images_dir, f"{num}_photo.jpg")
            dscope_path = os.path.join(images_dir, f"{num}_dscope.jpg")
            if not os.path.isfile(photo_path) or not os.path.isfile(dscope_path):
                skipped += 1
                continue
            split_cache[num] = compute_split_ratio(photo_path, dscope_path)

        ratio = split_cache[num]
        clin_json = str(row.get("viz_grounding_clinical", "[]"))
        dscope_json = str(row.get("viz_grounding_dscope", "[]"))

        df.at[idx, "viz_grounding_clinical_remapped"] = remap_entries(clin_json, ratio, "clinical")
        df.at[idx, "viz_grounding_dscope_remapped"] = remap_entries(dscope_json, ratio, "dscope")
        remapped += 1

    df.to_csv(csv_path, index=False)
    print(f"  Remapped: {remapped}, Skipped: {skipped}")
    print(f"  Written: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remap dscope/clinical grounding boxes to combined-image coordinates")
    parser.add_argument("--csv", type=str, default=None,
                        help="Process only this model name (e.g. 'dermato_llama')")
    parser.add_argument("--images-dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    csvs = find_viz_csvs(args.results_dir, filter_model=args.csv)
    if not csvs:
        print(f"No *_predictions_reason_viz.csv found in {args.results_dir}")
        sys.exit(1)
    print(f"Found {len(csvs)} CSV(s): {[m for m, _ in csvs]}")

    for model_name, csv_path in csvs:
        process_csv(model_name, csv_path, args.images_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
