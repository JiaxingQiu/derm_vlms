#!/usr/bin/env python
"""Stage 1 (no GPU): read existing predictions, parse top-1 diagnosis.

    python prelim_simedit/run_preedit.py --robot medgemma --n 10
    python prelim_simedit/run_preedit.py --robot medgemma --case-ids 1_combined 5_combined 10_combined
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pipeline import run_preedit  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="medgemma")
    p.add_argument("--n", type=int, default=None, help="num cases (omit for all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="explicit case_id list (overrides --n/--seed)")
    p.add_argument("--image-mode", default="combined")
    args = p.parse_args()

    run_preedit(
        robot=args.robot,
        n=args.n,
        seed=args.seed,
        case_ids=args.case_ids,
        image_mode=args.image_mode,
    )


if __name__ == "__main__":
    main()
