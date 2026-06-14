#!/usr/bin/env python
"""Stage 3 (GPU): robot re-diagnoses given the judge's feedback.

    python prelim_simedit/run_postedit.py --robot medgemma --judge gpt53
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pipeline import run_postedit  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="medgemma")
    p.add_argument("--judge", default="gpt53")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    run_postedit(robot=args.robot, judge_name=args.judge,
                 max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
