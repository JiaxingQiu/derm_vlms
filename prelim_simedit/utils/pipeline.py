"""Stage functions for the three-phase editing-cycle simulation.

Each stage reads the previous stage's CSV and is resumable.

    run_preedit(robot, n, ...)       any   -> 01_preedit__<d>.csv
    run_judge(robot, judge, ...)     API   -> 02_judge__<judge>__<d>.csv
    run_postedit(robot, judge, ...)  GPU   -> 03_postedit__<judge>__<d>.csv
    run_eval(robot, judge, ...)      any   -> scored__ + summary__ csv
"""

import functools

import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import io
from .data import INPUT_COLS, build_inputs
from .eval import parse_top1, parse_top3_names, score, summarize
from .prompts import postedit_prompt

print = functools.partial(print, flush=True)

PREEDIT_COLS = INPUT_COLS + ["preedit_dx"]

JUDGE_COLS_TOP1 = PREEDIT_COLS + ["judge_dx", "judge_reasoning"]
JUDGE_COLS_TOP3 = PREEDIT_COLS + ["judge_corrected_differential"]

POSTEDIT_COLS_TOP1 = JUDGE_COLS_TOP1 + ["postedit_response", "postedit_dx"]
POSTEDIT_COLS_TOP3 = JUDGE_COLS_TOP3 + ["postedit_response", "postedit_dx"]

CHECKPOINT_EVERY = 5


def _judge_cols(differential):
    return JUDGE_COLS_TOP1 if differential == "top_1" else JUDGE_COLS_TOP3


def _postedit_cols(differential):
    return POSTEDIT_COLS_TOP1 if differential == "top_1" else POSTEDIT_COLS_TOP3


# --- Stage 1: preedit (NO GPU — reads existing predictions) -----------------

def run_preedit(robot, n=None, seed=42, case_ids=None, image_mode="combined",
                differential="top_1"):
    """Pull the existing response, parse diagnosis, and write the preedit CSV."""
    robot_name = robot if isinstance(robot, str) else robot.name
    out_path = io.stage_path("preedit", robot_name, differential=differential)

    df = build_inputs(robot_name, n=n, seed=seed, case_ids=case_ids,
                      image_mode=image_mode)

    rows = []
    for _, r in df.iterrows():
        row = {c: r[c] for c in INPUT_COLS}
        if differential == "top_1":
            row["preedit_dx"] = parse_top1(r["preedit_response"])
        else:
            row["preedit_dx"] = str(r["preedit_response"]).strip()
        rows.append(row)

    result = pd.DataFrame(rows, columns=PREEDIT_COLS)
    io.write_df(out_path, result)
    print(f"[preedit/{robot_name}/{differential}] {len(result)} cases -> {out_path}")
    return result


# --- Stage 2: judge (API, no GPU) -------------------------------------------

def run_judge(robot_name, judge, differential="top_1",
              checkpoint_every=CHECKPOINT_EVERY):
    judge = _as_judge(judge)
    in_path = io.stage_path("preedit", robot_name, differential=differential)
    out_path = io.stage_path("judge", robot_name, judge.name,
                             differential=differential)
    judge_cols = _judge_cols(differential)

    df_in = pd.read_csv(in_path)
    done = io.done_ids(out_path)
    pending = df_in[~df_in["case_id"].astype(str).isin(done)]
    print(f"[judge/{robot_name}/{judge.name}/{differential}] "
          f"{len(pending)}/{len(df_in)} pending")
    if pending.empty:
        return pd.read_csv(out_path)

    judge.ensure_loaded()
    batch = []
    for _, r in tqdm(pending.iterrows(), total=len(pending)):
        judge_kwargs = {"differential": differential}
        if judge.name == "ground_truth":
            judge_kwargs["gt_y16"] = r.get("gt_y16", "")

        result = _safe(
            lambda: judge.judge(
                Image.open(r["image_path"]).convert("RGB"),
                r["preedit_dx"], **judge_kwargs),
            r["case_id"], default={})

        row = {c: r[c] for c in PREEDIT_COLS}
        if not isinstance(result, dict):
            result = {}

        if differential == "top_1":
            row["judge_dx"] = result.get("diagnosis", "")
            row["judge_reasoning"] = result.get("reasoning", "")
        else:
            row["judge_corrected_differential"] = result.get(
                "corrected_differential", "")

        batch.append(row)
        if len(batch) >= checkpoint_every:
            io.append_rows(out_path, batch, judge_cols)
            batch = []
    io.append_rows(out_path, batch, judge_cols)
    print(f"[judge/{robot_name}/{judge.name}/{differential}] -> {out_path}")
    return pd.read_csv(out_path)


# --- Stage 3: postedit (GPU) ------------------------------------------------

def run_postedit(robot, judge_name, differential="top_1",
                 max_new_tokens=None, checkpoint_every=CHECKPOINT_EVERY):
    if max_new_tokens is None:
        max_new_tokens = 64 if differential == "top_1" else 512

    robot = _as_robot(robot)
    in_path = io.stage_path("judge", robot.name, judge_name,
                            differential=differential)
    out_path = io.stage_path("postedit", robot.name, judge_name,
                             differential=differential)
    judge_cols = _judge_cols(differential)
    postedit_cols = _postedit_cols(differential)

    df_in = pd.read_csv(in_path)
    done = io.done_ids(out_path)
    pending = df_in[~df_in["case_id"].astype(str).isin(done)]
    print(f"[postedit/{robot.name}/{judge_name}/{differential}] "
          f"{len(pending)}/{len(df_in)} pending")
    if pending.empty:
        return pd.read_csv(out_path)

    robot.ensure_loaded()
    batch = []
    for _, r in tqdm(pending.iterrows(), total=len(pending)):
        if differential == "top_1":
            prompt = postedit_prompt(
                differential="top_1",
                judge_dx=r.get("judge_dx", ""),
                reasoning=r.get("judge_reasoning", ""),
            )
        else:
            prompt = postedit_prompt(
                differential="top_3",
                corrected_differential=r.get(
                    "judge_corrected_differential", ""),
            )

        resp = _safe(lambda: robot.predict(
            Image.open(r["image_path"]).convert("RGB"),
            prompt, max_new_tokens), r["case_id"])

        row = {c: r[c] for c in judge_cols}
        row["postedit_response"] = resp
        if differential == "top_1":
            row["postedit_dx"] = parse_top1(resp)
        else:
            row["postedit_dx"] = str(resp).strip() if resp else ""
        batch.append(row)
        if len(batch) >= checkpoint_every:
            io.append_rows(out_path, batch, postedit_cols)
            batch = []
    io.append_rows(out_path, batch, postedit_cols)
    print(f"[postedit/{robot.name}/{judge_name}/{differential}] -> {out_path}")
    return pd.read_csv(out_path)


# --- Stage 4: eval ----------------------------------------------------------

def run_eval(robot_name, judge_name, differential="top_1"):
    in_path = io.stage_path("postedit", robot_name, judge_name,
                            differential=differential)
    scored_df = score(pd.read_csv(in_path), differential=differential)
    summary = summarize(scored_df, robot=robot_name, judge=judge_name,
                        differential=differential)

    scored_path = io.stage_path("scored", robot_name, judge_name,
                                differential=differential)
    summary_path = io.stage_path("summary", robot_name, judge_name,
                                 differential=differential)
    io.write_df(scored_path, scored_df)
    io.write_df(summary_path, summary)
    print(f"[eval/{robot_name}/{judge_name}/{differential}] -> {scored_path}")
    print(summary.to_string(index=False))
    return scored_df, summary


# --- helpers ----------------------------------------------------------------

def _as_robot(robot):
    if isinstance(robot, str):
        from .robots import get_robot
        return get_robot(robot)
    return robot


def _as_judge(judge):
    if isinstance(judge, str):
        from .judges import get_judge
        return get_judge(judge)
    return judge


def _safe(fn, case_id, default="ERROR"):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {case_id}: {e}")
        return f"ERROR: {e}" if default == "ERROR" else default
