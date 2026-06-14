# prelim_simedit — AI-agent editing-cycle simulation

Replaces the human reviewer with an AI judge in a 3-phase loop:

```
[preedit]   weak robot (HF VLM) → top-1 diagnosis
[judge]     strong judge (API VLM) → {correct/incorrect, correct_dx, reasoning}
[postedit]  weak robot + feedback → revised top-1 diagnosis
[eval]      pre/judge/post accuracy vs MIDAS ground truth (y16)
```

---

## Folder layout

```
prelim_simedit/
  utils/
    data.py        build_inputs(n): sample combined cases, prep images, attach GT
    prompts.py     PREEDIT / JUDGE / POSTEDIT templates
    io.py          stage paths + checkpoint/resume helpers
    eval.py        parse→y16 (reuses prelim_acc), per-phase + judge metrics
    pipeline.py    run_preedit / run_judge / run_postedit / run_eval
    robots/        pluggable medrobots (local HF VLMs)
      base.py, medgemma.py, dermato_llama.py, __init__.py
    judges/        pluggable judges (API VLMs)
      base.py, gpt53.py, __init__.py
  run_preedit.py   CLI stage 1 (GPU)
  run_judge.py     CLI stage 2 (API, no GPU)
  run_postedit.py  CLI stage 3 (GPU)
  run_eval.py      CLI stage 4 (any node)
  notebooks/
    run_pipeline.ipynb   runs all 4 stages inline for N examples
  results_local/         self-contained outputs
    images/              prepared combined images for sampled cases
    01_preedit__<robot>.csv
    02_judge__<robot>__<judge>.csv
    03_postedit__<robot>__<judge>.csv
    scored__<robot>__<judge>.csv
    summary__<robot>__<judge>.csv
```

---

## Self-containment guarantees

- **Reads only** from outside this folder (never writes):
  - `data_share/midas_share.parquet` (ground truth)
  - `data_share/case_mapping.parquet` (case_id ↔ lesion_id)
  - `data/*.jpg` (source lesion images)
  - `tokens.py` (API keys)
  - `collect_ai_response/<model>/utils.py` (model loaders, imported as module)
  - `prelim_acc/{parse,match}.py` (diagnosis parsing + y16 matching)
- **All outputs** go to `prelim_simedit/results_local/`

---

## Usage (notebook)

Run on a GPU node with the `dermato_llama` conda env:

```python
# In run_pipeline.ipynb — set N, ROBOT, JUDGE at top, run all cells
ROBOT = "medgemma"   # or "dermato_llama"
JUDGE = "gpt53"
N = 5
```

## Usage (CLI, per-phase)

```bash
# Phase 1 (GPU)
python prelim_simedit/run_preedit.py --robot medgemma --n 10

# Phase 2 (API, any node)
python prelim_simedit/run_judge.py --robot medgemma --judge gpt53

# Phase 3 (GPU)
python prelim_simedit/run_postedit.py --robot medgemma --judge gpt53

# Eval (any node)
python prelim_simedit/run_eval.py --robot medgemma --judge gpt53
```

Each stage is resumable (skips case_ids already in output CSV).

---

## Adding a new robot or judge

**Robot** (local HF VLM):
1. Create `utils/robots/<name>.py` subclassing `Robot`
2. Implement `load()` and `predict(image, prompt, max_new_tokens)`
3. Add to `ROBOT_REGISTRY` in `utils/robots/__init__.py`

**Judge** (API VLM):
1. Create `utils/judges/<name>.py` subclassing `Judge`
2. Implement `load()` and `judge(image, dx)`
3. Add to `JUDGE_REGISTRY` in `utils/judges/__init__.py`
