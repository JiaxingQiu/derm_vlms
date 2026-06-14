"""Stage artifact paths + checkpoint/resume helpers.

Each stage reads the previous stage's CSV and appends its own rows, so every
stage is idempotent and resumable (skip ``case_id``s already present).
"""

import os
from pathlib import Path

import pandas as pd

from .paths import RESULTS_LOCAL

# Override with set_output_dir() to route outputs (e.g. to results_local/test/)
_output_root = None


def set_output_dir(path):
    """Override the output root directory (e.g. for notebook test runs).

    Pass None to reset to default (results_local/<robot>/).
    """
    global _output_root
    _output_root = Path(path) if path is not None else None


def _get_robot_dir(robot):
    root = _output_root if _output_root is not None else RESULTS_LOCAL / robot
    root.mkdir(parents=True, exist_ok=True)
    return root


def stage_path(stage, robot, judge=None):
    """Return the CSV path for a given stage.

    stage in {"preedit", "judge", "postedit", "scored", "summary"}.
    Results go to the override dir (if set) or results_local/<robot>/.
    """
    robot_dir = _get_robot_dir(robot)
    names = {
        "preedit": "01_preedit.csv",
        "judge": f"02_judge__{judge}.csv",
        "postedit": f"03_postedit__{judge}.csv",
        "scored": f"scored__{judge}.csv",
        "summary": f"summary__{judge}.csv",
    }
    return str(robot_dir / names[stage])


def done_ids(path, key="case_id"):
    """Return the set of already-completed ids in an output CSV (for resume)."""
    if os.path.exists(path):
        return set(pd.read_csv(path)[key].astype(str).tolist())
    return set()


def append_rows(path, rows, col_order):
    """Append a list of dict rows to a CSV, writing the header only once."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[col_order]
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def write_df(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
