#!/usr/bin/env python
"""Run visual grounding on all *_predictions_reason.csv files.

For every row in each input CSV, this script:
  1. Parses the reasoning sentences from `reason_classify`.
  2. Loads the corresponding image.
  3. Asks Qwen3-VL-8B to locate the image region for each sentence.
  4. Saves a new CSV  <modelname>_predictions_reason_viz.csv  with all
     original columns plus a `viz_grounding` column (JSON list of
     {sentence, diagnosis, box} dicts).

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
    if raw_path and raw_path != "None" and os.path.isfile(raw_path):
        return raw_path
    if raw_path and raw_path != "None":
        fname = os.path.basename(raw_path)
        candidate = os.path.join(images_dir, fname)
        if os.path.isfile(candidate):
            return candidate
    return None


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

    # Resume support
    done_ids: set = set()
    if os.path.exists(out_path):
        done_df = pd.read_csv(out_path)
        done_ids = set(done_df["id"].tolist())
        print(f"Resuming — {len(done_ids)} rows already done")

    pending = df[~df["id"].isin(done_ids)]
    if limit:
        pending = pending.head(limit)
    print(f"Processing {len(pending)} rows")

    if pending.empty:
        print("Nothing to do.")
        return

    batch: list[dict] = []

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc=model_name):
        img_path = resolve_image_path(str(row.get("image_path", "")), images_dir)
        reason_text = str(row.get("reason_classify", ""))

        if not img_path:
            viz_json = "[]"
        else:
            try:
                image = Image.open(img_path).convert("RGB")
                results = ground_reasoning_sentences(model, processor, image, reason_text)
                viz_json = grounding_results_to_json(results)
            except Exception as e:
                print(f"[ERR] {row['id']}: {e}")
                viz_json = json.dumps([{"error": str(e)}])

        out_row = row.to_dict()
        out_row["viz_grounding"] = viz_json
        batch.append(out_row)

        if len(batch) >= CHECKPOINT_EVERY:
            _checkpoint(batch, out_path)
            batch = []
            gc.collect()
            torch.cuda.empty_cache()

    if batch:
        _checkpoint(batch, out_path, final=True)

    result_df = pd.read_csv(out_path)
    print(f"Done. {len(result_df)} total rows written to {out_path}")


def _checkpoint(batch: list[dict], out_path: str, final: bool = False):
    batch_df = pd.DataFrame(batch)
    header = not os.path.exists(out_path)
    batch_df.to_csv(out_path, mode="a", header=header, index=False)
    tag = "Final" if final else "Checkpoint"
    print(f"[{tag}] +{len(batch)} rows")


def main():
    parser = argparse.ArgumentParser(description="Visual grounding with Qwen3-VL-8B")
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

    print("Loading Qwen3-VL-8B-Instruct...")
    torch.cuda.empty_cache()
    model, processor = load_model(hf_token=hf_token)
    print("Model ready.\n")

    for model_name, csv_path in csvs:
        process_csv(model_name, csv_path, model, processor,
                    images_dir=args.images_dir, limit=args.limit)

    print("\nAll done.")


if __name__ == "__main__":
    main()
