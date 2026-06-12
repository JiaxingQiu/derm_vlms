"""Derive case_id ↔ lesion_id mapping from midas_share.parquet.

Rule: case_num = rank of lesion_id, ordered by its earliest uid.
      case_id  = {case_num}_{image_mode}

Usage:
    python data_utils/generate_case_mapping.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data_share" / "midas_share.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data_share" / "case_mapping.parquet"

IMAGE_MODES = ["photo", "dscope", "combined", "virtual"]


def generate_case_mapping(parquet_path=PARQUET_PATH, output_path=OUTPUT_PATH):
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("uid")

    # Determine which image modes each lesion has
    lesion_modes = df.groupby("lesion_id")["lesion_distance"].apply(set)

    # Rank lesions by first uid appearance
    lesion_order = (
        df.groupby("lesion_id")["uid"]
        .min()
        .sort_values()
        .index
    )

    rows = []
    for case_num, lesion_id in enumerate(lesion_order, start=1):
        distances = lesion_modes[lesion_id]

        has_photo = "6in" in distances or "1ft" in distances
        has_dscope = "dscope" in distances
        has_virtual = "virtual" in distances
        has_combined = has_photo and has_dscope

        for mode, available in [
            ("photo", has_photo),
            ("dscope", has_dscope),
            ("combined", has_combined),
            ("virtual", has_virtual),
        ]:
            if available:
                rows.append({
                    "case_id": f"{case_num}_{mode}",
                    "lesion_id": lesion_id,
                    "image_mode": mode,
                })

    mapping = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(output_path, index=False)

    n_lesions = mapping["lesion_id"].nunique()
    print(f"Generated {output_path}")
    print(f"  {len(mapping)} case_ids from {n_lesions} lesions")
    print(f"  Modes: {mapping['image_mode'].value_counts().to_dict()}")

    return mapping


if __name__ == "__main__":
    generate_case_mapping()
