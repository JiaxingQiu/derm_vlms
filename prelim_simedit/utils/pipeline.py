"""Stage functions for the three-phase editing-cycle simulation.

Each stage reads the previous stage's CSV and is resumable. The four functions
here are what both the CLI runners and the notebook call.

    run_preedit(robot, n, ...)       any   -> 01_preedit__<robot>.csv  (reads existing CSV)
    run_judge(robot, judge)          API   -> 02_judge__<robot>__<judge>.csv
    run_postedit(robot, judge)       GPU   -> 03_postedit__<robot>__<judge>.csv
    run_eval(robot, judge)           any   -> scored__ + summary__ csv
"""

import functools

import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import io
from .data import INPUT_COLS, build_inputs
from .eval import parse_top1, score, summarize
from .prompts import postedit_prompt

print = functools.partial(print, flush=True)

PREEDIT_COLS = INPUT_COLS + ["preedit_dx"]
JUDGE_COLS = PREEDIT_COLS + [
    "judge_verdict", "judge_correct_dx", "judge_reasoning", "judge_raw",
]
POSTEDIT_COLS = JUDGE_COLS + ["postedit_response", "postedit_dx"]

CHECKPOINT_EVERY = 5


# --- Stage 1: preedit (NO GPU — reads existing predictions) -----------------

def run_preedit(robot, n=None, seed=42, case_ids=None, image_mode="combined"):
    """Pull the existing top-3 response, parse top-1, and write the preedit CSV.

    Args:
        robot: robot name (str), e.g. "medgemma" or "dermato_llama"
        n: number of cases to sample (None = all)
        seed: random seed for sampling
        case_ids: explicit list of case_id strings (overrides n/seed)
    """
    robot_name = robot if isinstance(robot, str) else robot.name
    out_path = io.stage_path("preedit", robot_name)

    df = build_inputs(robot_name, n=n, seed=seed, case_ids=case_ids,
                      image_mode=image_mode)

    rows = []
    for _, r in df.iterrows():
        row = {c: r[c] for c in INPUT_COLS}
        row["preedit_dx"] = parse_top1(r["preedit_response"])
        rows.append(row)

    result = pd.DataFrame(rows, columns=PREEDIT_COLS)
    io.write_df(out_path, result)
    print(f"[preedit/{robot_name}] {len(result)} cases -> {out_path}")
    return result


# --- Stage 2: judge (Azure API, no GPU) -------------------------------------

def run_judge(robot_name, judge, checkpoint_every=CHECKPOINT_EVERY):
    judge = _as_judge(judge)
    in_path = io.stage_path("preedit", robot_name)
    out_path = io.stage_path("judge", robot_name, judge.name)

    df_in = pd.read_csv(in_path)
    done = io.done_ids(out_path)
    pending = df_in[~df_in["case_id"].astype(str).isin(done)]
    print(f"[judge/{robot_name}/{judge.name}] {len(pending)}/{len(df_in)} pending")
    if pending.empty:
        return pd.read_csv(out_path)

    judge.ensure_loaded()
    batch = []
    for _, r in tqdm(pending.iterrows(), total=len(pending)):
        verdict = _safe(lambda: judge.judge(Image.open(r["image_path"]).convert("RGB"),
                                            r["preedit_dx"]), r["case_id"], default={})
        row = {c: r[c] for c in PREEDIT_COLS}
        row["judge_verdict"] = verdict.get("verdict", "") if isinstance(verdict, dict) else ""
        row["judge_correct_dx"] = verdict.get("correct_diagnosis", "") if isinstance(verdict, dict) else ""
        row["judge_reasoning"] = verdict.get("reasoning", "") if isinstance(verdict, dict) else ""
        row["judge_raw"] = verdict.get("raw", "") if isinstance(verdict, dict) else str(verdict)
        batch.append(row)
        if len(batch) >= checkpoint_every:
            io.append_rows(out_path, batch, JUDGE_COLS)
            batch = []
    io.append_rows(out_path, batch, JUDGE_COLS)
    print(f"[judge/{robot_name}/{judge.name}] -> {out_path}")
    return pd.read_csv(out_path)


# --- Stage 3: postedit (GPU) ------------------------------------------------

def run_postedit(robot, judge_name, max_new_tokens=64,
                 checkpoint_every=CHECKPOINT_EVERY):
    robot = _as_robot(robot)
    in_path = io.stage_path("judge", robot.name, judge_name)
    out_path = io.stage_path("postedit", robot.name, judge_name)

    df_in = pd.read_csv(in_path)
    done = io.done_ids(out_path)
    pending = df_in[~df_in["case_id"].astype(str).isin(done)]
    print(f"[postedit/{robot.name}/{judge_name}] {len(pending)}/{len(df_in)} pending")
    if pending.empty:
        return pd.read_csv(out_path)

    robot.ensure_loaded()
    batch = []
    for _, r in tqdm(pending.iterrows(), total=len(pending)):
        prompt = postedit_prompt(
            preedit_dx=r["preedit_dx"],
            verdict=r.get("judge_verdict", ""),
            correct_dx=r.get("judge_correct_dx", ""),
            reasoning=r.get("judge_reasoning", ""),
        )
        resp = _safe(lambda: robot.predict(Image.open(r["image_path"]).convert("RGB"),
                                           prompt, max_new_tokens), r["case_id"])
        row = {c: r[c] for c in JUDGE_COLS}
        row["postedit_response"] = resp
        row["postedit_dx"] = parse_top1(resp)
        batch.append(row)
        if len(batch) >= checkpoint_every:
            io.append_rows(out_path, batch, POSTEDIT_COLS)
            batch = []
    io.append_rows(out_path, batch, POSTEDIT_COLS)
    print(f"[postedit/{robot.name}/{judge_name}] -> {out_path}")
    return pd.read_csv(out_path)


# --- Stage 4: eval (any node) -----------------------------------------------

def run_eval(robot_name, judge_name):
    in_path = io.stage_path("postedit", robot_name, judge_name)
    scored_df = score(pd.read_csv(in_path))
    summary = summarize(scored_df, robot=robot_name, judge=judge_name)

    scored_path = io.stage_path("scored", robot_name, judge_name)
    summary_path = io.stage_path("summary", robot_name, judge_name)
    io.write_df(scored_path, scored_df)
    io.write_df(summary_path, summary)
    print(f"[eval/{robot_name}/{judge_name}] -> {scored_path}")
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
