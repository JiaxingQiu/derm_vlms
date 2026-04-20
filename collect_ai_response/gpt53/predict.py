#!/usr/bin/env python
"""GPT-5.3 prediction on all MIDAS lesions with checkpointing."""

import os
import sys
import functools
print = functools.partial(print, flush=True)

PROJECT_ROOT = "/scratch/jq2uw/derm_vlms"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "gpt53"))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from PIL import Image
from tqdm import tqdm
from tokens import AZURE_GPT53_API_KEY
from utils import init_client, predict_image
from data_utils import prepare_all_lesions

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
OUT_PATH = os.path.join(RESULTS_DIR, "gpt53_predictions_all.csv")

PROMPT = (
    "You are an expert dermatologist. Please describe using dermatological "
    "terms to describe the appearance of this lesion. "
    "Please give the top 3 diagnoses in your differential."
)
CHECKPOINT_EVERY = 10
COL_ORDER = [
    "id", "ground_truth", "y16", "y16_description", "image_mode",
    "describe_then_classify", "image_path", "original_image_name", "lesion_id",
]


def main():
    print("Initialising Azure GPT-5.3 client...")
    client = init_client(api_key=AZURE_GPT53_API_KEY)
    print("Client ready.")

    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data_share", "midas_share.parquet"))
    print(f"Loaded {len(df)} rows")

    df_all = prepare_all_lesions(df, data_dir=DATA_DIR, output_dir=IMAGES_DIR)
    print(f"Total prediction rows: {len(df_all)}")

    # Resume
    if os.path.exists(OUT_PATH):
        done_ids = set(pd.read_csv(OUT_PATH)["id"].tolist())
        print(f"Resuming: {len(done_ids)} predictions already completed")
    else:
        done_ids = set()

    pending = df_all[~df_all["id"].isin(done_ids)]
    print(f"{len(pending)} predictions remaining")

    if pending.empty:
        print("Nothing to do.")
        return

    batch = []
    lesions_in_batch = set()

    for _, row in tqdm(pending.iterrows(), total=len(pending)):
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"[SKIP] {row['id']}: {e}")
            continue

        try:
            response = predict_image(client, image, prompt=PROMPT)
        except Exception as e:
            print(f"[ERR] {row['id']}: {e}")
            response = f"ERROR: {e}"

        batch.append({
            "id": row["id"],
            "ground_truth": row["ground_truth"],
            "y16": row["y16"],
            "y16_description": row["y16_description"],
            "image_mode": row["image_mode"],
            "describe_then_classify": response,
            "image_path": row["image_path"],
            "original_image_name": row["original_image_name"],
            "lesion_id": row["lesion_id"],
        })

        lesions_in_batch.add(row["lesion_id"])

        if len(lesions_in_batch) >= CHECKPOINT_EVERY:
            _checkpoint(batch, lesions_in_batch)
            batch = []
            lesions_in_batch = set()

    if batch:
        _checkpoint(batch, lesions_in_batch, final=True)

    results_df = pd.read_csv(OUT_PATH)
    print(f"\nDone. Total predictions: {len(results_df)}")
    print(f"Unique lesions: {results_df['lesion_id'].nunique()}")
    print(f"Image modes:\n{results_df['image_mode'].value_counts()}")


def _checkpoint(batch, lesions_in_batch, final=False):
    batch_df = pd.DataFrame(batch)[COL_ORDER]
    header = not os.path.exists(OUT_PATH)
    batch_df.to_csv(OUT_PATH, mode="a", header=header, index=False)
    tag = "Final" if final else "Checkpoint"
    print(f"\n[{tag}] +{len(batch)} rows ({len(lesions_in_batch)} lesions)")


if __name__ == "__main__":
    main()
