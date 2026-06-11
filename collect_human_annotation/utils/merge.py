"""Utilities for loading the admin CSV export and merging with source data.

Typical usage in a notebook:

    from res_eng.utils import load_merged
    df = load_merged("path/to/annotations_export.csv")
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "results"
DEFAULT_MIDAS_PATH = PROJECT_ROOT / "data_share" / "midas_share.parquet"


# ---------------------------------------------------------------------------
# Step 1: Load the admin CSV export
# ---------------------------------------------------------------------------

def load_admin_export(csv_path):
    """Load the admin-exported annotations CSV and parse structured columns.

    Returns a DataFrame with:
      - case_id, model, login_id, evaluator profile columns
      - diag_{1,2,3}_name, diag_{1,2,3}_label, diag_{1,2,3}_correct_differential
      - parsed reasoning, diagnosis_order, total_duration_seconds
      - derived: case_num (int) and image_mode (str) extracted from case_id
    """
    df = pd.read_csv(csv_path)

    # Parse case_id into numeric index and image mode
    parts = df["case_id"].str.extract(r"^(\d+)_(.+)$")
    df["case_num"] = parts[0].astype(int)
    df["image_mode"] = parts[1]

    # Parse JSON columns safely
    for col in ["diagnosis_order", "page_visits"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_json_parse)

    return df


# ---------------------------------------------------------------------------
# Step 2: Build case_id → lesion metadata lookup from prediction CSVs
# ---------------------------------------------------------------------------

def load_predictions_lookup(results_dir=None):
    """Build a lookup from case_id to lesion_id + ground truth.

    Reads prediction CSVs from results_dir and deduplicates to one row
    per case_id. Any single model CSV suffices since lesion_id is shared.

    Returns DataFrame with columns:
      id (== case_id), lesion_id, ground_truth, y16, y16_description, image_mode
    """
    results_dir = Path(results_dir) if results_dir else DEFAULT_PREDICTIONS_DIR

    csvs = sorted(results_dir.glob("*_predictions_reason.csv"))
    if not csvs:
        raise FileNotFoundError(f"No prediction CSVs found in {results_dir}")

    # Use first available CSV — lesion_id mapping is identical across models
    df = pd.read_csv(csvs[0])
    keep = ["id", "lesion_id", "ground_truth", "y16", "y16_description", "image_mode"]
    return df[keep].drop_duplicates(subset="id")


# ---------------------------------------------------------------------------
# Step 3: Join to MIDAS parquet for full clinical metadata
# ---------------------------------------------------------------------------

def merge_to_midas(annotations_df, predictions_lookup, midas_path=None):
    """Merge annotations with predictions lookup and MIDAS clinical data.

    Returns a single flat DataFrame ready for analysis.
    """
    midas_path = Path(midas_path) if midas_path else DEFAULT_MIDAS_PATH
    midas = pd.read_parquet(midas_path)

    # One row per lesion for joining (repeated rows differ only by image distance)
    midas_lesion = midas.drop_duplicates(subset="lesion_id")

    # Join annotations → predictions (adds lesion_id, ground_truth, y16)
    merged = annotations_df.merge(
        predictions_lookup,
        left_on="case_id",
        right_on="id",
        how="left",
        suffixes=("", "_pred"),
    )

    # Join → MIDAS (adds demographics, skin type, location, pathology, etc.)
    midas_cols = [
        "lesion_id",
        "id_patient",
        "demo_gender",
        "demo_age",
        "demo_fitzpatrick_skintype",
        "x_skintype",
        "x_skincolor",
        "x_skintone",
        "x_location",
        "lesion_location",
        "lesion_length_mm",
        "lesion_width_mm",
        "notes_clinical_impression_1",
        "notes_pathreport",
    ]
    available_cols = [c for c in midas_cols if c in midas_lesion.columns]
    merged = merged.merge(
        midas_lesion[available_cols],
        on="lesion_id",
        how="left",
    )

    return merged


# ---------------------------------------------------------------------------
# Convenience: one-call pipeline
# ---------------------------------------------------------------------------

def load_merged(csv_path, results_dir=None, midas_path=None):
    """End-to-end: load admin CSV → join predictions → join MIDAS.

    Args:
        csv_path: Path to the admin-exported annotations CSV.
        results_dir: Directory containing *_predictions_reason.csv files.
                     Defaults to <project_root>/results/
        midas_path: Path to midas_share.parquet.
                    Defaults to <project_root>/data_share/midas_share.parquet

    Returns:
        Analysis-ready DataFrame with all annotation, prediction, and
        clinical metadata columns.
    """
    annotations = load_admin_export(csv_path)
    predictions = load_predictions_lookup(results_dir)
    return merge_to_midas(annotations, predictions, midas_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_parse(val):
    if pd.isna(val) or val == "":
        return None
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val
