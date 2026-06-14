"""Build the experiment input batch from existing prediction CSVs.

Phase 1 (preedit) already ran in the main pipeline — the top-3 differential
+ reasoning is stored in ``results/<robot>_predictions_reason.csv``.  We just
read it (combined mode), optionally subset, and parse out the top-1 diagnosis.

This means Phase 1 needs NO GPU.  Only Phase 3 (postedit) needs a GPU to
re-query the robot with the judge's feedback.

All images referenced by ``image_path`` in the CSV already exist under
``results/images/``.  We copy the subset into our self-contained
``results_local/images/`` so downstream stages don't depend on the root
results folder at runtime.
"""

import shutil
from pathlib import Path

import pandas as pd

from .paths import PROJECT_ROOT, RESULTS_LOCAL

RESULTS_DIR = PROJECT_ROOT / "results"

ROBOT_CSV = {
    "medgemma": RESULTS_DIR / "medgemma_predictions_reason.csv",
    "dermato_llama": RESULTS_DIR / "dermato_llama_predictions_reason.csv",
}

INPUT_COLS = [
    "case_id", "lesion_id", "image_mode", "image_path",
    "gt_y16", "gt_y16_description", "gt_y3",
    "preedit_response",
]


def build_inputs(robot, n=None, seed=42, case_ids=None, image_mode="combined"):
    """Load existing predictions and prepare the preedit batch.

    Filtering (applied in order):
      1. ``image_mode`` (default: combined)
      2. ``case_ids`` — explicit list of case_id strings (overrides n)
      3. ``n`` — first n cases sorted by case_id (if case_ids is None)
      If both ``case_ids`` and ``n`` are None, uses ALL combined cases.

    Returns a DataFrame with columns INPUT_COLS.
    """
    csv_path = ROBOT_CSV.get(robot)
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError(
            f"No predictions CSV for robot '{robot}'. "
            f"Expected: {csv_path}\nAvailable: {list(ROBOT_CSV)}"
        )

    df = pd.read_csv(csv_path)
    df = df[df["image_mode"] == image_mode].copy()
    df["y16"] = df["y16"].fillna("Other")
    # Normalize non-standard GT labels to "Other"
    _STANDARD_Y16 = {
        "Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma",
        "Fibrous Papule", "Hemangioma", "Melanocytic Lesion",
        "Melanocytic Nevus", "Melanocytic Tumor", "Melanoma",
        "Seborrheic Keratosis", "Squamous Cell Carcinoma",
        "Squamous Cell Carcinoma In Situ",
    }
    df["y16"] = df["y16"].apply(lambda x: x if x in _STANDARD_Y16 else "Other")
    df = df.rename(columns={"id": "case_id"})

    # Sort by numeric prefix so "1_combined" < "2_combined" < "10_combined"
    df["_sort_key"] = df["case_id"].str.extract(r"^(\d+)").astype(int)
    df = df.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    if case_ids is not None:
        case_ids = [str(c) for c in case_ids]
        df = df[df["case_id"].isin(case_ids)]
    elif n is not None and n < len(df):
        df = df.head(n)

    from .io import _get_robot_dir
    images_dir = _get_robot_dir(robot) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        src = Path(r["image_path"])
        if not src.is_file():
            src = RESULTS_DIR / "images" / src.name
        if not src.is_file():
            print(f"[SKIP] {r['case_id']}: image not found at {src}")
            continue

        dst = images_dir / src.name
        if not dst.is_file():
            shutil.copy2(src, dst)

        rows.append({
            "case_id": r["case_id"],
            "lesion_id": r["lesion_id"],
            "image_mode": r["image_mode"],
            "image_path": str(dst),
            "gt_y16": r["y16"],
            "gt_y16_description": r["y16_description"],
            "gt_y3": r["ground_truth"],
            "preedit_response": r["reason_classify"],
        })

    out = pd.DataFrame(rows, columns=INPUT_COLS)
    print(f"build_inputs({robot}): {len(out)} {image_mode} cases")
    return out
