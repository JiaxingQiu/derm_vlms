#!/usr/bin/env python
"""Stage 2 (Azure API): judge reviews the robot's diagnosis from the image.

    python prelim_simedit/run_judge.py --robot medgemma --judge gpt53
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pipeline import run_judge  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="medgemma")
    p.add_argument("--judge", default="gpt53")
    args = p.parse_args()

    run_judge(robot_name=args.robot, judge=args.judge)


if __name__ == "__main__":
    main()
