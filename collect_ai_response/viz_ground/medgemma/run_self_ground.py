#!/usr/bin/env python
"""Run MedGemma self-grounding on its predictions CSV.

Asks MedGemma to draw bounding boxes for its OWN reasoning sentences.
Output has the same column structure as the Qwen3 external grounding CSV,
written to a SEPARATE file:

    medgemma_predictions_reason_viz_self.csv

Existing _viz.csv files (Qwen3 external grounding) are NEVER touched.

Usage:
    python run_self_ground.py                    # full run
    python run_self_ground.py --limit 20         # first 20 rows (debug)
"""

import argparse
import functools
import gc
import json
import os
import sys

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from util import (
    load_model,
    parse_reasoning_sentences,
    predict_grounding_box,
    grounding_results_to_json,
)
from tokens import HF_TOKEN

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
CHECKPOINT_EVERY = 10
MODEL_NAME = "medgemma"


def resolve_image_path(raw_path: str, images_dir: str) -> str | None:
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
    if variant == "dscope":
        return combined_path.replace("_combined.", "_dscope.")
    elif variant == "clinical":
        return combined_path.replace("_combined.", "_photo.")
    return combined_path


def _is_row_done(row: pd.Series) -> bool:
    if pd.isna(row.get("viz_grounding")) or row.get("viz_grounding") == "":
        return False
    if row.get("image_mode") == "combined":
        if pd.isna(row.get("viz_grounding_dscope")) or row.get("viz_grounding_dscope") == "":
            return False
        if pd.isna(row.get("viz_grounding_clinical")) or row.get("viz_grounding_clinical") == "":
            return False
    return True


def _ground_on_image(model, processor, img_path: str, reason_text: str) -> str:
    image = Image.open(img_path).convert("RGB")
    parsed = parse_reasoning_sentences(reason_text)
    results = []
    for entry in parsed:
        box, raw = predict_grounding_box(model, processor, image, entry["text"])
        results.append({
            "type": entry["type"],
            "diagnosis": entry["diagnosis"],
            "text": entry["text"],
            "box": box,
        })
    return json.dumps(results, separators=(",", ":"))


def process(model, processor, csv_path: str, images_dir: str, limit: int | None = None):
    out_path = csv_path.replace("_predictions_reason.csv", "_predictions_reason_viz_self.csv")

    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME} (self-grounding)")
    print(f"Input:  {csv_path}  ({len(df)} rows)")
    print(f"Output: {out_path}")

    if os.path.exists(out_path):
        out_df = pd.read_csv(out_path)
        for col in ("viz_grounding", "viz_grounding_dscope", "viz_grounding_clinical"):
            if col not in out_df.columns:
                out_df[col] = ""
        print(f"Resuming from existing: {len(out_df)} rows")
    else:
        out_df = pd.DataFrame()

    done_map: dict[str, dict] = {}
    if not out_df.empty:
        for _, r in out_df.iterrows():
            done_map[r["id"]] = r.to_dict()

    pending_rows = []
    for _, row in df.iterrows():
        row_id = row["id"]
        existing = done_map.get(row_id)
        if existing and _is_row_done(pd.Series(existing)):
            continue
        pending_rows.append((row, existing))

    if limit:
        pending_rows = pending_rows[:limit]

    print(f"Pending: {len(pending_rows)} rows")

    if not pending_rows:
        print("Nothing to do.")
        return

    updates = 0
    for row, existing in tqdm(pending_rows, desc=f"{MODEL_NAME} self-ground"):
        row_id = row["id"]
        reason_text = str(row.get("reason_classify", ""))
        is_combined = row.get("image_mode") == "combined"

        out_row = existing.copy() if existing else row.to_dict()

        needs_main = pd.isna(out_row.get("viz_grounding")) or out_row.get("viz_grounding") in ("", None)
        if needs_main:
            img_path = resolve_image_path(str(row.get("image_path", "")), images_dir)
            if not img_path:
                out_row["viz_grounding"] = "[]"
            else:
                out_row["viz_grounding"] = _ground_on_image(model, processor, img_path, reason_text)

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
                        out_row[col] = _ground_on_image(model, processor, variant_path, reason_text)
        else:
            out_row.setdefault("viz_grounding_dscope", "")
            out_row.setdefault("viz_grounding_clinical", "")

        done_map[row_id] = out_row
        updates += 1

        if updates % CHECKPOINT_EVERY == 0:
            _write_full(done_map, df, out_path)
            gc.collect()
            torch.cuda.empty_cache()

    _write_full(done_map, df, out_path)
    _remap_combined_rows(out_path, images_dir)
    print(f"Done. Written to {out_path}")


def _remap_combined_rows(out_path: str, images_dir: str):
    """Add remapped columns for combined rows (same logic as remap_boxes_to_combined.py)."""
    df = pd.read_csv(out_path)
    combined = df[df["image_mode"] == "combined"]
    if combined.empty:
        return

    if "viz_grounding_clinical_remapped" not in df.columns:
        df["viz_grounding_clinical_remapped"] = ""
        df["viz_grounding_dscope_remapped"] = ""

    split_cache: dict[str, float] = {}
    remapped = 0

    for idx, row in combined.iterrows():
        case_id = row["id"]
        num = case_id.replace("_combined", "")

        if num not in split_cache:
            photo_path = os.path.join(images_dir, f"{num}_photo.jpg")
            dscope_path = os.path.join(images_dir, f"{num}_dscope.jpg")
            if not os.path.isfile(photo_path) or not os.path.isfile(dscope_path):
                continue
            photo = Image.open(photo_path)
            dscope = Image.open(dscope_path)
            h = min(photo.height, dscope.height)
            pw = int(photo.width * h / photo.height)
            dw = int(dscope.width * h / dscope.height)
            split_cache[num] = pw / (pw + dw)

        ratio = split_cache.get(num)
        if ratio is None:
            continue

        clin_json = str(row.get("viz_grounding_clinical", "[]"))
        dscope_json = str(row.get("viz_grounding_dscope", "[]"))

        df.at[idx, "viz_grounding_clinical_remapped"] = _remap_entries(clin_json, ratio, "clinical")
        df.at[idx, "viz_grounding_dscope_remapped"] = _remap_entries(dscope_json, ratio, "dscope")
        remapped += 1

    df.to_csv(out_path, index=False)
    print(f"[Remap] {remapped} combined rows remapped")


def _remap_entries(entries_json: str, split_ratio: float, side: str) -> str:
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
            box = entry["box"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            if side == "clinical":
                out["box"] = {"x": round(x * split_ratio, 4), "y": round(y, 4),
                              "w": round(w * split_ratio, 4), "h": round(h, 4)}
            else:
                out["box"] = {"x": round(split_ratio + x * (1 - split_ratio), 4), "y": round(y, 4),
                              "w": round(w * (1 - split_ratio), 4), "h": round(h, 4)}
        remapped.append(out)
    return json.dumps(remapped, separators=(",", ":"))


def _write_full(done_map: dict[str, dict], input_df: pd.DataFrame, out_path: str):
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
    parser = argparse.ArgumentParser(description="MedGemma self-grounding")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--images-dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or HF_TOKEN

    csv_path = os.path.join(args.results_dir, f"{MODEL_NAME}_predictions_reason.csv")
    if not os.path.isfile(csv_path):
        print(f"Not found: {csv_path}")
        sys.exit(1)

    print("Loading MedGemma...")
    model, processor = load_model(hf_token=hf_token)
    print("Model ready.\n")

    process(model, processor, csv_path, args.images_dir, limit=args.limit)
    print("\nAll done.")


if __name__ == "__main__":
    main()
