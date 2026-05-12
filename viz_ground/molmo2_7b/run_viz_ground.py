#!/usr/bin/env python
"""Run visual grounding on all *_predictions_reason.csv files using Molmo2-O-7B.

For every row in each input CSV, this script:
  1. Parses the reasoning sentences from `reason_classify`.
  2. Loads the corresponding image.
  3. Asks Molmo2-O-7B to locate the image region for each sentence.
  4. Saves a new CSV  <modelname>_predictions_reason_viz.csv  with all
     original columns plus grounding columns.

Grounding columns:
  - viz_grounding              : boxes from the row's own image
  - viz_grounding_dscope       : (combined rows only) boxes from dscope image
  - viz_grounding_clinical     : (combined rows only) boxes from clinical photo

Resume logic: a row is "done" when all expected grounding columns are populated.
Combined rows missing dscope/clinical columns are treated as partially done and
only the missing groundings are computed.

Usage:
    python run_viz_ground.py                        # process all CSVs
    python run_viz_ground.py --csv gpt53            # only gpt53
    python run_viz_ground.py --limit 20             # first 20 rows (debug)
    python run_viz_ground.py --images-dir /alt/path # override image dir
"""

import argparse
import functools
import gc
import glob
import json
import os
import sys

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from util import (
    ground_reasoning_sentences,
    grounding_results_to_json,
    load_model,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
CHECKPOINT_EVERY = 10


def find_input_csvs(results_dir: str, filter_model: str | None = None) -> list[tuple[str, str]]:
    """Return list of (model_name, csv_path) for each *_predictions_reason.csv."""
    pattern = os.path.join(results_dir, "*_predictions_reason.csv")
    pairs = []
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        model_name = basename.replace("_predictions_reason.csv", "")
        if filter_model and model_name != filter_model:
            continue
        pairs.append((model_name, path))
    return pairs


def resolve_image_path(raw_path: str, images_dir: str) -> str | None:
    """Try to locate the image file, return path or None."""
    if not raw_path or raw_path == "None":
        return None
    if os.path.isfile(raw_path):
        return raw_path
    fname = os.path.basename(raw_path)
    candidate = os.path.join(images_dir, fname)
    if os.path.isfile(candidate):
        return candidate
    return None


def _variant_image_path(combined_path: str, variant: str) -> str:
    """Derive dscope or clinical path from a combined image path."""
    if variant == "dscope":
        return combined_path.replace("_combined.", "_dscope.")
    elif variant == "clinical":
        return combined_path.replace("_combined.", "_photo.")
    return combined_path


def _is_row_done(row: pd.Series) -> bool:
    """Check if a row has all expected grounding columns populated."""
    if pd.isna(row.get("viz_grounding")) or row.get("viz_grounding") == "":
        return False
    if row.get("image_mode") == "combined":
        if pd.isna(row.get("viz_grounding_dscope")) or row.get("viz_grounding_dscope") == "":
            return False
        if pd.isna(row.get("viz_grounding_clinical")) or row.get("viz_grounding_clinical") == "":
            return False
    return True


def _ground_on_image(model, processor, img_path: str, reason_text: str) -> str:
    """Run grounding on a single image, return JSON string."""
    image = Image.open(img_path).convert("RGB")
    results = ground_reasoning_sentences(model, processor, image, reason_text)
    return grounding_results_to_json(results)


def process_csv(model_name: str, csv_path: str, model, processor,
                images_dir: str, limit: int | None = None):
    """Process one CSV and write the _viz output."""
    out_path = os.path.join(
        os.path.dirname(csv_path),
        f"{model_name}_predictions_reason_viz.csv",
    )

    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Input:  {csv_path}  ({len(df)} rows)")
    print(f"Output: {out_path}")

    # Load existing results for resume
    if os.path.exists(out_path):
        out_df = pd.read_csv(out_path)
        for col in ("viz_grounding", "viz_grounding_dscope", "viz_grounding_clinical"):
            if col not in out_df.columns:
                out_df[col] = ""
        print(f"Loaded existing output: {len(out_df)} rows")
    else:
        out_df = pd.DataFrame()

    # Build lookup of existing results by id
    done_map: dict[str, dict] = {}
    if not out_df.empty:
        for _, r in out_df.iterrows():
            done_map[r["id"]] = r.to_dict()

    # Determine which rows still need work
    pending_rows = []
    for _, row in df.iterrows():
        row_id = row["id"]
        existing = done_map.get(row_id)
        if existing and _is_row_done(pd.Series(existing)):
            continue
        pending_rows.append((row, existing))

    if limit:
        pending_rows = pending_rows[:limit]

    n_combined = sum(1 for row, _ in pending_rows if row.get("image_mode") == "combined")
    print(f"Pending: {len(pending_rows)} rows ({n_combined} combined, "
          f"{len(pending_rows) - n_combined} non-combined)")

    if not pending_rows:
        print("Nothing to do.")
        return

    updates: list[dict] = []

    for row, existing in tqdm(pending_rows, desc=model_name):
        row_id = row["id"]
        reason_text = str(row.get("reason_classify", ""))
        is_combined = row.get("image_mode") == "combined"

        # Start from existing data or fresh row
        out_row = existing.copy() if existing else row.to_dict()

        # --- viz_grounding (own image) ---
        needs_main = pd.isna(out_row.get("viz_grounding")) or out_row.get("viz_grounding") in ("", None)
        if needs_main:
            img_path = resolve_image_path(str(row.get("image_path", "")), images_dir)
            if not img_path:
                out_row["viz_grounding"] = "[]"
            else:
                try:
                    out_row["viz_grounding"] = _ground_on_image(model, processor, img_path, reason_text)
                except Exception as e:
                    print(f"[ERR] {row_id} (main): {e}")
                    out_row["viz_grounding"] = json.dumps([{"error": str(e)}])

        # --- viz_grounding_dscope and viz_grounding_clinical (combined only) ---
        if is_combined:
            combined_img_path = resolve_image_path(str(row.get("image_path", "")), images_dir)

            for variant, col in [("dscope", "viz_grounding_dscope"), ("clinical", "viz_grounding_clinical")]:
                needs_variant = pd.isna(out_row.get(col)) or out_row.get(col) in ("", None)
                if not needs_variant:
                    continue
                if not combined_img_path:
                    out_row[col] = "[]"
                else:
                    variant_path = _variant_image_path(combined_img_path, variant)
                    if not os.path.isfile(variant_path):
                        out_row[col] = "[]"
                    else:
                        try:
                            out_row[col] = _ground_on_image(model, processor, variant_path, reason_text)
                        except Exception as e:
                            print(f"[ERR] {row_id} ({variant}): {e}")
                            out_row[col] = json.dumps([{"error": str(e)}])
        else:
            out_row.setdefault("viz_grounding_dscope", "")
            out_row.setdefault("viz_grounding_clinical", "")

        done_map[row_id] = out_row
        updates.append(out_row)

        if len(updates) >= CHECKPOINT_EVERY:
            _write_full(done_map, df, out_path)
            updates = []
            gc.collect()
            torch.cuda.empty_cache()

    if updates:
        _write_full(done_map, df, out_path)

    result_df = pd.read_csv(out_path)
    print(f"Done. {len(result_df)} total rows written to {out_path}")


def _write_full(done_map: dict[str, dict], input_df: pd.DataFrame, out_path: str):
    """Write all completed rows to disk, preserving input row order."""
    rows = []
    for _, row in input_df.iterrows():
        row_id = row["id"]
        if row_id in done_map:
            rows.append(done_map[row_id])
    if rows:
        out_df = pd.DataFrame(rows)
        col_order = [c for c in input_df.columns if c in out_df.columns]
        for extra in ("viz_grounding", "viz_grounding_dscope", "viz_grounding_clinical"):
            if extra not in col_order and extra in out_df.columns:
                col_order.append(extra)
        out_df = out_df[col_order]
        out_df.to_csv(out_path, index=False)
        print(f"[Checkpoint] {len(out_df)} rows written")


def main():
    parser = argparse.ArgumentParser(description="Visual grounding with Molmo2-O-7B")
    parser.add_argument("--csv", type=str, default=None,
                        help="Process only this model name (e.g. 'gpt53')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to process per CSV (for debugging)")
    parser.add_argument("--images-dir", type=str, default=IMAGES_DIR,
                        help="Directory containing images")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help="Directory containing input CSVs")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    csvs = find_input_csvs(args.results_dir, filter_model=args.csv)
    if not csvs:
        print(f"No *_predictions_reason.csv files found in {args.results_dir}")
        sys.exit(1)
    print(f"Found {len(csvs)} CSV(s): {[m for m, _ in csvs]}")

    print("Loading Molmo2-O-7B...")
    torch.cuda.empty_cache()
    model, processor = load_model(hf_token=hf_token)
    print("Model ready.\n")

    for model_name, csv_path in csvs:
        process_csv(model_name, csv_path, model, processor,
                    images_dir=args.images_dir, limit=args.limit)

    print("\nAll done.")


if __name__ == "__main__":
    main()
