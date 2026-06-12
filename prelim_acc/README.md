# Preliminary Accuracy

Automated evaluation of VLM predictions against ground truth (no human review).

---

| # | Script | What it does | Input | Output |
|---|--------|-------------|-------|--------|
| 1 | `parse.py` | Extract top-3 diagnosis names from free-text model responses | raw response text | list of diagnosis strings |
| 2 | `match.py` | Map free-text diagnoses to canonical 16 `y16` labels via synonym dictionary | diagnosis strings | matched `y16` labels |
| 3 | `metrics.py` | Compute top-1 / top-3 accuracy, breakdowns by class, image mode, etc. | parsed + matched df | accuracy tables |
| 4 | `notebooks/eval_summary.ipynb` | Run full evaluation pipeline on all models | `results/*_predictions_all.csv` | summary tables + figures |

---

**Usage:**

```python
from prelim_acc.metrics import eval_model, compute_metrics

df = pd.read_csv("results/medgemma_predictions_all.csv")
df = eval_model(df)          # parse responses, match to y16, compute correctness
results = compute_metrics(df)  # overall, by_image_mode, by_y3, by_y16, unmatched
```
