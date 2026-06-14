"""Parse diagnoses to canonical y16 labels and compute phase accuracies.

Reuses the existing ``prelim_acc`` parser + synonym matcher so the simulation
scores diagnoses exactly like the rest of the project.

Supports two modes:
  - top_1: single diagnosis accuracy
  - top_3: top-3 hit-rate (correct if GT is anywhere in top-3)
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
    """Extract a single top-1 diagnosis string from a robot response."""
    if not text or not isinstance(text, str):
        return ""
    m = _DX_LINE.search(text)
    if m:
        dx = m.group(1).strip().strip("*").rstrip(".")
        if len(dx) > 3 and dx.lower() != "diagnosis":
            return dx
    m = _NUMBERED_BOLD.search(text)
    if m:
        dx = m.group(1).strip().rstrip(".")
        if len(dx) > 3:
            return dx
    m = _NUMBERED_PLAIN.search(text)
    if m:
        dx = m.group(1).strip().rstrip(".")
        if len(dx) > 3 and dx.lower() != "diagnosis":
            return dx
    top = extract_top3(text)
    if top and top[0].lower() != "diagnosis":
        return top[0]
    stripped = text.strip().splitlines()[0].strip().rstrip(".")
    if stripped and len(stripped) < 100 and stripped.lower() != "diagnosis":
        return stripped
    return ""


def parse_top3_names(text):
    """Extract up to 3 diagnosis names from a response."""
    if not text or not isinstance(text, str):
        return []
    top = extract_top3(text)
    filtered = [d for d in top if d and d.lower() != "diagnosis"][:3]
    if filtered:
        return filtered
    items = _parse_numbered_items(text)
    return items[:3]


_NUMBERED_ITEM = re.compile(
    r"^\s*\d+[\.\)]\s*\*{0,2}([^*:\n]+?)(?:\*{0,2}\s*[:\-\(]|\*{0,2}\s*$)",
    re.MULTILINE,
)


def _parse_numbered_items(text):
    matches = _NUMBERED_ITEM.findall(text)
    return [m.strip().rstrip(".") for m in matches if m.strip() and len(m.strip()) > 2]


def to_y16(dx_text):
    """Map a free-text diagnosis to a y16 label. Unmapped → 'Other'."""
    if not dx_text or not isinstance(dx_text, str):
        return "Other"
    if dx_text.strip().lower() == "other":
        return "Other"
    mapped = match_to_y16(dx_text)
    return mapped if mapped else "Other"


def _top3_hit(dx_list, gt_y16):
    """Return True if GT is in any of the mapped y16 labels from dx_list."""
    if not dx_list:
        return False
    return any(to_y16(dx) == gt_y16 for dx in dx_list)


def score(df, differential="top_1"):
    """Measure accuracy at each stage (preedit, judge, postedit) against gt_y16."""
    df = df.copy()

    if differential == "top_1":
        if "preedit_dx" not in df.columns or df["preedit_dx"].isna().any():
            df["preedit_dx"] = df["preedit_response"].apply(parse_top1)
        if "postedit_dx" not in df.columns or df["postedit_dx"].isna().any():
            df["postedit_dx"] = df["postedit_response"].apply(parse_top1)

        df["preedit_y16"] = df["preedit_dx"].apply(to_y16)
        df["postedit_y16"] = df["postedit_dx"].apply(to_y16)
        df["judge_y16"] = df["judge_dx"].apply(to_y16)

        df["preedit_correct"] = df["preedit_y16"] == df["gt_y16"]
        df["postedit_correct"] = df["postedit_y16"] == df["gt_y16"]
        df["judge_correct"] = df["judge_y16"] == df["gt_y16"]

    else:  # top_3
        df["preedit_top3"] = df["preedit_dx"].apply(parse_top3_names)
        df["postedit_top3"] = df["postedit_dx"].apply(parse_top3_names)

        df["preedit_dx1"] = df["preedit_top3"].apply(
            lambda lst: lst[0] if lst else "")
        df["postedit_dx1"] = df["postedit_top3"].apply(
            lambda lst: lst[0] if lst else "")

        df["preedit_y16_top1"] = df["preedit_dx1"].apply(to_y16)
        df["postedit_y16_top1"] = df["postedit_dx1"].apply(to_y16)

        df["preedit_top1_correct"] = df["preedit_y16_top1"] == df["gt_y16"]
        df["postedit_top1_correct"] = df["postedit_y16_top1"] == df["gt_y16"]

        df["preedit_top3_correct"] = df.apply(
            lambda r: _top3_hit(r["preedit_top3"], r["gt_y16"]), axis=1)
        df["postedit_top3_correct"] = df.apply(
            lambda r: _top3_hit(r["postedit_top3"], r["gt_y16"]), axis=1)

        if "judge_corrected_differential" in df.columns:
            df["judge_top3"] = df["judge_corrected_differential"].apply(
                parse_top3_names)
            df["judge_dx1"] = df["judge_top3"].apply(
                lambda lst: lst[0] if lst else "")
            df["judge_top1_y16"] = df["judge_dx1"].apply(to_y16)
            df["judge_top1_correct"] = df["judge_top1_y16"] == df["gt_y16"]
            df["judge_top3_correct"] = df.apply(
                lambda r: _top3_hit(r["judge_top3"], r["gt_y16"]), axis=1)

        df["preedit_correct"] = df["preedit_top1_correct"]
        df["postedit_correct"] = df["postedit_top1_correct"]

    return df


def _acc(series):
    return float(series.mean()) if len(series) else float("nan")


def summarize(scored_df, robot=None, judge=None, differential="top_1"):
    """One-row summary: accuracy at each stage vs GT."""
    n = len(scored_df)
    improved = (~scored_df["preedit_correct"] & scored_df["postedit_correct"]).sum()
    regressed = (scored_df["preedit_correct"] & ~scored_df["postedit_correct"]).sum()

    row = {
        "robot": robot,
        "judge": judge,
        "differential": differential,
        "n": n,
    }

    if differential == "top_1":
        row.update({
            "preedit_acc": _acc(scored_df["preedit_correct"]),
            "judge_acc": _acc(scored_df["judge_correct"]),
            "postedit_acc": _acc(scored_df["postedit_correct"]),
            "delta": _acc(scored_df["postedit_correct"]) - _acc(scored_df["preedit_correct"]),
            "n_improved": int(improved),
            "n_regressed": int(regressed),
        })
    else:  # top_3
        row.update({
            "preedit_top1": _acc(scored_df["preedit_top1_correct"]),
            "preedit_top3": _acc(scored_df["preedit_top3_correct"]),
        })
        if "judge_top1_correct" in scored_df.columns:
            row["judge_top1"] = _acc(scored_df["judge_top1_correct"])
            row["judge_top3"] = _acc(scored_df["judge_top3_correct"])
        row.update({
            "postedit_top1": _acc(scored_df["postedit_top1_correct"]),
            "postedit_top3": _acc(scored_df["postedit_top3_correct"]),
            "delta_top1": _acc(scored_df["postedit_top1_correct"]) - _acc(scored_df["preedit_top1_correct"]),
            "delta_top3": _acc(scored_df["postedit_top3_correct"]) - _acc(scored_df["preedit_top3_correct"]),
            "n_improved": int(improved),
            "n_regressed": int(regressed),
        })

    return pd.DataFrame([row])
