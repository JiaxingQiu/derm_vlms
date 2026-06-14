#!/usr/bin/env python
"""Stage 4 (any node): score pre/judge/post phases against ground truth.

    python prelim_simedit/run_eval.py --robot medgemma --judge gpt53
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pipeline import run_eval  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="medgemma")
    p.add_argument("--judge", default="gpt53")
    p.add_argument("--differential", default="top_1", choices=["top_1", "top_3"])
    args = p.parse_args()

    run_eval(robot_name=args.robot, judge_name=args.judge,
             differential=args.differential)


if __name__ == "__main__":
    main()
