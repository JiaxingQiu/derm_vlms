#!/usr/bin/env python
"""Combine external (Qwen3-8B) and self-grounded viz CSVs per model.

For each model, sorts unique case numbers, then takes self-grounded boxes
for the first half and external (Qwen3) boxes for the second half.

Output: results/<model>_predictions_reason_viz_combined.csv

Usage:
    python combine_viz_sources.py              # all 3 models
    python combine_viz_sources.py --model gpt53  # one model
    python combine_viz_sources.py --dry-run    # preview split without writing
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODELS = {
    "gpt53": "gpt53_predictions_reason",
    "medgemma": "medgemma_predictions_reason",
    "dermato_llama": "dermato_llama_predictions_reason",
}

VIZ_COLS = [
    "viz_grounding",
    "viz_grounding_dscope",
    "viz_grounding_clinical",
    "viz_grounding_clinical_remapped",
    "viz_grounding_dscope_remapped",
]


def case_number(case_id: str) -> int:
    """Extract the numeric prefix from a case ID like '1025_combined'."""
    return int(case_id.rsplit("_", 1)[0])


def combine_model(model_key: str, dry_run: bool = False):
    base = MODELS[model_key]
    ext_path = os.path.join(RESULTS_DIR, f"{base}_viz.csv")
    self_path = os.path.join(RESULTS_DIR, f"{base}_viz_self.csv")
    out_path = os.path.join(RESULTS_DIR, f"{base}_viz_combined.csv")

    for p, label in [(ext_path, "external"), (self_path, "self")]:
        if not os.path.exists(p):
            print(f"[SKIP] {model_key}: {label} CSV not found: {p}")
            return

    df_ext = pd.read_csv(ext_path)
    df_self = pd.read_csv(self_path)

    assert list(df_ext.columns) == list(df_self.columns), (
        f"Column mismatch for {model_key}"
    )
    assert len(df_ext) == len(df_self), (
        f"Row count mismatch for {model_key}: ext={len(df_ext)} self={len(df_self)}"
    )

    case_nums = sorted(df_ext["id"].apply(case_number).unique())
    mid = len(case_nums) // 2
    self_cases = set(case_nums[:mid])

    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"  External: {ext_path}")
    print(f"  Self:     {self_path}")
    print(f"  Total case numbers: {len(case_nums)}")
    print(f"  Split: first {mid} -> self, last {len(case_nums) - mid} -> external")
    print(f"  Self  range: {case_nums[0]} .. {case_nums[mid - 1]}")
    print(f"  Ext   range: {case_nums[mid]} .. {case_nums[-1]}")

    use_self = df_ext["id"].apply(lambda x: case_number(x) in self_cases)
    n_self_rows = use_self.sum()
    n_ext_rows = len(df_ext) - n_self_rows
    print(f"  Rows from self: {n_self_rows},  rows from external: {n_ext_rows}")

    if dry_run:
        print("  [DRY RUN] Not writing output.")
        return

    # Start from the external CSV (all non-viz columns are identical),
    # then overwrite viz columns for self-grounded rows.
    df_out = df_ext.copy()
    for col in VIZ_COLS:
        df_out.loc[use_self, col] = df_self.loc[use_self, col].values

    df_out["qwen3_box"] = (~use_self).astype(int)

    df_out.to_csv(out_path, index=False)
    print(f"  Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Process only this model (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the split without writing files.")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS.keys())
    for m in models:
        combine_model(m, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
