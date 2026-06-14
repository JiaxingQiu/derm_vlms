# prelim_simedit — Simulated Editing-Cycle Experiment

Replaces the human reviewer with an **AI judge** to simulate a full editing cycle at scale, measuring how much a weaker diagnostic model improves when given expert-level feedback.

---

## Three-Phase Pipeline

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Phase 1:   │         │  Phase 2:   │         │  Phase 3:   │
│  PRE-EDIT   │───────▶ │    JUDGE    │───────▶ │  POST-EDIT  │
│ (Weak VLM)  │         │(Strong LLM) │         │ (Weak VLM)  │
└─────────────┘         └─────────────┘         └─────────────┘
    Robot gives             Judge reviews           Robot re-diagnoses
    initial dx              & corrects              with feedback
```


| Phase        | Actor           | Task                                              | Compute            |
| ------------ | --------------- | ------------------------------------------------- | ------------------ |
| 1. Pre-edit  | Weak robot      | Initial differential from lesion image            | GPU (already done) |
| 2. Judge     | Strong judge    | Per-diagnosis verdict + correction + reasoning    | API only           |
| 3. Post-edit | Same weak robot | Re-diagnose with judge's feedback in prompt       | GPU                |
| 4. Eval      | —               | Score all phases against MIDAS ground truth (y16) | CPU                |


---

## Two Modes


|                | Top-1                                     | Top-3 (Differential)                                         |
| -------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Robot output   | Single diagnosis                          | 3 diagnoses + reasoning                                      |
| Judge task     | One verdict + correction                  | Per-diagnosis verdict + correction                           |
| Post-edit hint | "X is not correct, answer is Y because Z" | "These are not correct: A, B, C. Corrected: 1. X: reason..." |
| Eval metrics   | top-1 accuracy                            | top-1 + top-3 hit-rate                                       |


---

## Prompts (Top-3 Mode Example)

**Phase 1 — Robot:**

```
You are an expert dermatologist. Please give the top 3 diagnoses in
your differential, and provide reasoning for each diagnosis, in format:
1. [Diagnosis]: [Reasoning]
2. [Diagnosis]: [Reasoning]
3. [Diagnosis]: [Reasoning]
```

**Phase 2 — Judge** (sees image + robot's differential):

```
...review EACH diagnosis...For each, state whether it is correct or
incorrect, provide the correct diagnosis, and explain why.

→ JSON: {verdict_1, correct_diagnosis_1, reasoning_1, ...per dx}
```

**Phase 3 — Robot** (same question + appended feedback):

```
...give the top 3 diagnoses...
1. [Diagnosis]: [Reasoning]  ...

Note: A reviewing expert dermatologist has indicated that these
differentials are not correct: Melanocytic Nevus, BCC, SCC.
The expert's corrected differential is:
1. Basal Cell Carcinoma: pearly translucent border with telangiectasia
2. Melanoma: irregular pigment network and asymmetry
3. Squamous Cell Carcinoma: ulcerated nodule with rapid growth
```

---

## Models

**Weak Robots:** MedGemma 1.5 4B-IT, DermatoLlama 11B

**Strong Judges:** GPT-5.3, GPT-5.4, Claude Opus 4.8, Claude Opus 4.6, Claude Sonnet 4.6, Claude Fable 5

---

## Evaluation Metrics


| Metric                     | Top-1                       | Top-3                                    |
| -------------------------- | --------------------------- | ---------------------------------------- |
| Pre-edit accuracy          | `preedit_acc`               | `preedit_top1_acc`, `preedit_top3_acc`   |
| Post-edit accuracy         | `postedit_acc`              | `postedit_top1_acc`, `postedit_top3_acc` |
| Improvement                | `delta_acc`                 | `delta_top1`, `delta_top3`               |
| Judge accuracy             | `judge_dx_acc`              | `judge_top1_acc`, `judge_top3_acc`       |
| Verdict agreement          | `judge_verdict_agreement`   | same (overall)                           |
| Cases improved / regressed | `n_improved`, `n_regressed` | same                                     |


---

## Data

- **MIDAS dataset** — skin lesion images with biopsy-confirmed ground truth
- **Images**: Combined (clinical photo + dermatoscopy) per lesion
- **Labels**: y16 fine-grained diagnostic labels (16 categories); missing → "Other"

---

## Project Structure

```
prelim_simedit/
├── utils/
│   ├── prompts.py      # Prompt templates (top_1 & top_3)
│   ├── pipeline.py     # 4-stage orchestrator
│   ├── eval.py         # Parsing + y16 mapping + scoring
│   ├── data.py         # Load predictions, filter, prep images
│   ├── io.py           # Paths, checkpointing, resume
│   ├── robots/         # Plug-and-play: medgemma, dermato_llama
│   └── judges/         # Plug-and-play: gpt53, gpt54, claude_*
├── run_{preedit,judge,postedit,eval}.py   # CLI runners
├── notebooks/run_pipeline.ipynb           # Interactive notebook
└── results_local/<robot>/                 # Self-contained outputs
```

---

## Usage

**Notebook** (GPU node):

```python
ROBOT = "medgemma"       # or "dermato_llama"
JUDGE = "gpt53"          # gpt54, claude_opus48, claude_fable, ...
DIFFERENTIAL = "top_3"   # or "top_1"
N = 10
```

**CLI** (per-phase, resumable):

```bash
python prelim_simedit/run_preedit.py  --robot medgemma --n 10 --differential top_3
python prelim_simedit/run_judge.py    --robot medgemma --judge gpt53 --differential top_3
python prelim_simedit/run_postedit.py --robot medgemma --judge gpt53 --differential top_3
python prelim_simedit/run_eval.py     --robot medgemma --judge gpt53 --differential top_3
```

---

## Key Design Decisions

- **Self-contained** — all outputs in `results_local/`, no writes outside this folder
- **Staged** — each phase runs independently (GPU vs API vs CPU)
- **Resumable** — checkpoints every 5 cases; skips completed IDs
- **Plug-and-play** — new judge/robot = one file + registry entry
- **Phase 1 reuses existing predictions** — no GPU re-run needed

---

## Research Questions

1. Does AI feedback improve weak model accuracy? (delta > 0?)
2. Which judge model is best?
3. How accurate are the judges themselves?
4. Does the judge correctly identify when the robot is right vs wrong?
5. Top-1 vs Top-3: does differential feedback yield better improvement?
6. How often does feedback cause regression?

