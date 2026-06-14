"""Parse diagnoses to canonical y16 labels and compute phase accuracies.

Reuses the existing ``prelim_acc`` parser + synonym matcher so the simulation
scores diagnoses exactly like the rest of the project.
"""

import re

import pandas as pd

from .paths import PROJECT_ROOT  # noqa: F401  (ensures repo root on sys.path)
from prelim_acc.parse import extract_top3
from prelim_acc.match import match_to_y16

_DX_LINE = re.compile(r"(?im)^\s*(?:final\s+|corrected\s+)?diagnos[ie]s\s*[:\-]\s*(.+?)\s*$")

_NUMBERED_BOLD = re.compile(
    r"^\s*1[\.\)]\s*\*{0,2}([^*:\n]+?)\*{0,2}\s*(?:[:\(\-]|$)",
    re.MULTILINE,
)

_NUMBERED_PLAIN = re.compile(
    r"^\s*1[\.\)]\s+([^:\n]+?)(?:\s*[:\-]|\s*$)",
    re.MULTILINE,
)


def parse_top1(text):
    """Extract a single top-1 diagnosis string from a robot response.

    Handles: explicit 'Diagnosis: X' lines, numbered lists with/without
    markdown bold, and falls back to prelim_acc's extract_top3.
    """
    if not text or not isinstance(text, str):
        return ""
    # 1. Explicit "Diagnosis: X" line (our postedit prompt format)
    m = _DX_LINE.search(text)
    if m:
        dx = m.group(1).strip().strip("*").rstrip(".")
        if len(dx) > 3 and dx.lower() != "diagnosis":
            return dx
    # 2. First numbered item with bold: "1. **Basal Cell Carcinoma (BCC):**"
    m = _NUMBERED_BOLD.search(text)
    if m:
        dx = m.group(1).strip().rstrip(".")
        if len(dx) > 3:
            return dx
    # 3. First numbered item plain: "1. Basal Cell Carcinoma:"
    m = _NUMBERED_PLAIN.search(text)
    if m:
        dx = m.group(1).strip().rstrip(".")
        if len(dx) > 3 and dx.lower() != "diagnosis":
            return dx
    # 4. Fall back to prelim_acc parser
    top = extract_top3(text)
    if top and top[0].lower() != "diagnosis":
        return top[0]
    # 5. If it's a short single-line string, treat it as the diagnosis itself
    stripped = text.strip().splitlines()[0].strip().rstrip(".")
    if stripped and len(stripped) < 100 and stripped.lower() != "diagnosis":
        return stripped
    return ""


def to_y16(dx_text):
    """Map a free-text diagnosis to a y16 label (or None)."""
    if not dx_text or not isinstance(dx_text, str):
        return None
    return match_to_y16(dx_text)


def score(df):
    """Add parsed labels + correctness flags to a postedit-stage DataFrame.

    Expects columns: gt_y16, preedit_dx, postedit_dx (or postedit_response),
                     judge_verdict, judge_correct_dx.
    """
    df = df.copy()

    if "preedit_dx" not in df.columns or df["preedit_dx"].isna().any():
        df["preedit_dx"] = df["preedit_response"].apply(parse_top1)
    if "postedit_dx" not in df.columns or df["postedit_dx"].isna().any():
        df["postedit_dx"] = df["postedit_response"].apply(parse_top1)

    df["preedit_y16"] = df["preedit_dx"].apply(to_y16)
    df["postedit_y16"] = df["postedit_dx"].apply(to_y16)
    df["judge_dx_y16"] = df["judge_correct_dx"].apply(to_y16)

    df["preedit_correct"] = df["preedit_y16"] == df["gt_y16"]
    df["postedit_correct"] = df["postedit_y16"] == df["gt_y16"]
    df["judge_dx_correct"] = df["judge_dx_y16"] == df["gt_y16"]

    verdict_says_correct = df["judge_verdict"].astype(str).str.lower().eq("correct")
    # Did the judge's correct/incorrect verdict match reality?
    df["judge_verdict_agree"] = verdict_says_correct == df["preedit_correct"]
    return df


def _acc(series):
    return float(series.mean()) if len(series) else float("nan")


def summarize(scored_df, robot=None, judge=None):
    """Return a one-row summary DataFrame of phase + judge metrics."""
    n = len(scored_df)
    improved = (~scored_df["preedit_correct"] & scored_df["postedit_correct"]).sum()
    regressed = (scored_df["preedit_correct"] & ~scored_df["postedit_correct"]).sum()
    row = {
        "robot": robot,
        "judge": judge,
        "n": n,
        "preedit_acc": _acc(scored_df["preedit_correct"]),
        "postedit_acc": _acc(scored_df["postedit_correct"]),
        "delta_acc": _acc(scored_df["postedit_correct"]) - _acc(scored_df["preedit_correct"]),
        "n_improved": int(improved),
        "n_regressed": int(regressed),
        "judge_dx_acc": _acc(scored_df["judge_dx_correct"]),
        "judge_verdict_agreement": _acc(scored_df["judge_verdict_agree"]),
    }
    return pd.DataFrame([row])
