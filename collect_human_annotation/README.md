# Results Engineering

Post-collection processing and analysis of annotations from the review interface.

---

## 1. Data Layers

Four data sources, joined through shared keys:

```
annotations_export.csv          ← human evaluations (from Django admin)
        │  case_id
        ▼
*_predictions_reason.csv        ← model outputs + case_id → lesion_id mapping
        │  lesion_id
        ▼
midas_share.parquet             ← clinical metadata (demographics, pathology, etc.)

results/images/{case_id}.jpg    ← lesion images (photo, dscope, combined)
```


| Source          | Granularity                        | Key                              | Location                                   |
| --------------- | ---------------------------------- | -------------------------------- | ------------------------------------------ |
| Admin CSV       | 1 row per evaluator × case × model | `case_id` + `model` + `login_id` | Django admin export                        |
| Prediction CSVs | 1 row per lesion × image mode      | `id` (= `case_id`)               | `results/` or blob `datasets/revlm_dc/`    |
| MIDAS parquet   | 1 row per image file               | `lesion_id`                      | `data_share/midas_share.parquet`           |
| Images          | 1 file per case_id                 | filename                         | `results/images/` or blob `deploy/images/` |


---

## 2. Key Identifiers


| ID           | Format                           | Example      | Scope                                 |
| ------------ | -------------------------------- | ------------ | ------------------------------------- |
| `case_id`    | `{num}_{mode}`                   | `1_photo`    | One image condition for one lesion    |
| `lesion_id`  | `{patient}_{location}_{control}` | `1_chest_no` | One physical lesion (multiple images) |
| `id_patient` | integer string                   | `1`          | One patient (multiple lesions)        |


The numeric prefix in `case_id` is a sequential index over sorted unique `lesion_id` values.

---

## 3. Pipeline

### Step 1 — Export annotations

Django admin → Users → **Export CSV** → saves `annotations_export.csv`.

### Step 2 — Merge

```python
import sys; sys.path.insert(0, "/path/to/derm_vlms")
from res_eng.utils import load_merged

df = load_merged("annotations_export.csv")
```

Or step-by-step:

```python
from res_eng.utils import load_admin_export, load_predictions_lookup, merge_to_midas

annotations = load_admin_export("annotations_export.csv")
predictions = load_predictions_lookup()          # reads from results/
merged = merge_to_midas(annotations, predictions)  # joins MIDAS parquet
```

### Step 3 — Analyze

```python
# Top-1 agreement rate per model
merged.groupby("model")["diag_1_label"].apply(lambda s: (s == "correct").mean())

# Top-3 hit rate
merged["any_correct"] = (
    (merged["diag_1_label"] == "correct") |
    (merged["diag_2_label"] == "correct") |
    (merged["diag_3_label"] == "correct")
)
merged.groupby("model")["any_correct"].mean()

# By ground truth
merged.groupby(["model", "ground_truth"])["any_correct"].mean()

# By skin color
merged.groupby(["model", "x_skincolor"])["any_correct"].mean()

# Timing
merged.groupby("model")["total_duration_seconds"].describe()
```

---

## 4. Admin CSV Column Reference

**Evaluator:** `login_id`, `full_name`, `occupation`, `years_experience`, `institution`, `dermoscopy_experience`

**Case:** `case_id`, `model`, `raw_response`

**Diagnosis evaluation (×3):**

- `diag_{1,2,3}_name` — AI's diagnosis name
- `diag_{1,2,3}_label` — human verdict: `correct` | `incorrect` | (empty)
- `diag_{1,2,3}_correct_differential` — human's replacement (when incorrect)
- `reasoning_{1,2,3}` — JSON, per-sentence edits to AI reasoning

**Behavioral:**

- `diagnosis_order` — JSON, user reordering of top-3 (empty = accepted AI order)
- `other_feedback` — free-text
- `marked_complete` — boolean
- `total_duration_seconds` — time on page
- `page_visits` — JSON, visit timestamps

---

## 5. Folder Structure

```
res_eng/
├── README.md
└── utils/
    ├── __init__.py
    └── merge.py        ← load_admin_export, load_predictions_lookup, merge_to_midas, load_merged

```

