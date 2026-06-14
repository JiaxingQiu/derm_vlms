# prelim_simedit — Simulated Editing-Cycle Experiment

Replaces the human reviewer with an AI judge to simulate a full editing cycle, measuring how much a weaker diagnostic model improves when given expert feedback.

## Pipeline

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Phase 1:   │         │  Phase 2:   │         │  Phase 3:   │
│  PRE-EDIT   │───────▶ │    JUDGE    │───────▶ │  POST-EDIT  │
│ (Weak VLM)  │         │(Strong LLM) │         │ (Weak VLM)  │
└─────────────┘         └─────────────┘         └─────────────┘
```

1. **Preedit** — Weak robot gives initial diagnosis from lesion image (reuses existing predictions, no GPU)
2. **Judge** — Strong model provides its own diagnosis (API call)
3. **Postedit** — Robot re-diagnoses with judge's answer appended as hint (GPU)
4. **Eval** — Score each stage against MIDAS ground truth (y16)

## Modes

- **top_1** — single diagnosis per stage
- **top_3** — top-3 differential with reasoning (`1. [Dx]: [Reasoning]`)

## Models

| Role | Models |
|------|--------|
| Robots | MedGemma 1.5 4B-IT, DermatoLlama 11B |
| Judges | GPT-5.3, GPT-5.4, Claude Opus 4.8/4.6, Claude Sonnet 4.6, Claude Fable 5, `ground_truth` (oracle ceiling) |

## Eval Metrics

Each stage scored independently against GT:

| top_1 | top_3 |
|-------|-------|
| `preedit_acc` | `preedit_top1`, `preedit_top3` |
| `judge_acc` | `judge_top1`, `judge_top3` |
| `postedit_acc` | `postedit_top1`, `postedit_top3` |
| `delta` | `delta_top1`, `delta_top3` |

## Usage

```python
ROBOT = "medgemma"
JUDGE = "gpt53"           # or "ground_truth" for ceiling
DIFFERENTIAL = "top_3"
N = 10
```

```bash
python prelim_simedit/run_preedit.py  --robot medgemma --n 10 --differential top_3
python prelim_simedit/run_judge.py    --robot medgemma --judge gpt53 --differential top_3
python prelim_simedit/run_postedit.py --robot medgemma --judge gpt53 --differential top_3
python prelim_simedit/run_eval.py     --robot medgemma --judge gpt53 --differential top_3
```

Each stage is resumable (skips completed case_ids).
