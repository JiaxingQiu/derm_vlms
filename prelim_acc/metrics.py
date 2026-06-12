"""Compute top-1 / top-3 accuracy and breakdowns for VLM predictions.

Works on a DataFrame that has been enriched with parsed + matched
diagnosis columns by eval_model().
"""

import pandas as pd
from .parse import extract_top3
from .match import match_to_y16


def eval_model(df):
    """Add parsed diagnosis columns to a predictions DataFrame.

    Expects columns: describe_then_classify, y16.
    Adds columns: pred_raw (list[str]), pred_y16 (list[str|None]),
                  top1_correct, top3_correct.
    Returns the enriched DataFrame (copy).
    """
    df = df.copy()
    df["pred_raw"] = df["describe_then_classify"].apply(
        lambda x: extract_top3(str(x)) if pd.notna(x) else []
    )
    df["pred_y16"] = df["pred_raw"].apply(
        lambda names: [match_to_y16(n) for n in names]
    )
    df["top1_correct"] = df.apply(
        lambda r: (r["pred_y16"][0] == r["y16"]) if r["pred_y16"] else False,
        axis=1,
    )
    df["top3_correct"] = df.apply(
        lambda r: r["y16"] in r["pred_y16"] if r["pred_y16"] else False,
        axis=1,
    )
    df["parse_success"] = df["pred_raw"].apply(lambda x: len(x) > 0)
    df["match_success"] = df["pred_y16"].apply(
        lambda x: any(v is not None for v in x) if x else False
    )
    return df


def _acc(series):
    """Compute accuracy as fraction, returning NaN if empty."""
    if len(series) == 0:
        return float("nan")
    return series.mean()


def overall_accuracy(df):
    """Return a single-row DataFrame with overall top-1 and top-3 accuracy."""
    return pd.DataFrame([{
        "top1_acc": _acc(df["top1_correct"]),
        "top3_acc": _acc(df["top3_correct"]),
        "n": len(df),
        "parsed": df["parse_success"].sum(),
        "matched": df["match_success"].sum(),
    }])


def accuracy_by(df, group_col):
    """Return accuracy broken down by a grouping column."""
    rows = []
    for name, grp in df.groupby(group_col):
        rows.append({
            group_col: name,
            "top1_acc": _acc(grp["top1_correct"]),
            "top3_acc": _acc(grp["top3_correct"]),
            "n": len(grp),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def unmatched_terms(df):
    """Return a DataFrame of diagnosis terms that could not be matched to y16."""
    rows = []
    for _, r in df.iterrows():
        for raw, y16 in zip(r["pred_raw"], r["pred_y16"]):
            if y16 is None:
                rows.append({
                    "id": r["id"],
                    "ground_truth_y16": r["y16"],
                    "unmatched_term": raw,
                })
    return pd.DataFrame(rows)


def compute_metrics(df):
    """Run the full metrics suite. Returns a dict of DataFrames.

    Keys: overall, by_image_mode, by_y3, by_y16, unmatched
    """
    return {
        "overall": overall_accuracy(df),
        "by_image_mode": accuracy_by(df, "image_mode"),
        "by_y3": accuracy_by(df, "ground_truth"),
        "by_y16": accuracy_by(df, "y16"),
        "unmatched": unmatched_terms(df),
    }
