# Data Utils

Raw MIDAS data → shared parquet → case mapping.

---


| #   | Script                     | What it does                                     | Input                                            | Output                                                           | IDs created                    |
| --- | -------------------------- | ------------------------------------------------ | ------------------------------------------------ | ---------------------------------------------------------------- | ------------------------------ |
| 1   | `data_summary.ipynb`       | Explore raw MIDAS data                           | `data/release_midas.xlsx`                        | —                                                                | —                              |
| 2   | `data_process.ipynb`       | Process tabular data, assign `uid`, save parquet | `data/release_midas.xlsx` + `data/images/midas/` | `data_share/midas_share.parquet` + `midas_share_dictionary.json` | `uid` (row order), `lesion_id` |
| 3   | `generate_case_mapping.py` | Derive `case_id ↔ lesion_id` mapping             | `data_share/midas_share.parquet`                 | `data_share/case_mapping.parquet`                                | `case_id`                      |


**Supporting files:** `utils.py` (processing + image prep), `tabular2text.py` (tabular → text), `__init__.py` (exports)

---

## ID Derivation


| ID          | How generated                                            | Interpretable?                 |
| ----------- | -------------------------------------------------------- | ------------------------------ |
| `uid`       | Row index + 1 in parquet (frozen)                        | Yes — canonical ordering key   |
| `lesion_id` | `= {midas_record_id}_{midas_location}_{midas_iscontrol}` | Yes — patient + site + control |
| `case_id`   | `= {rank_by_min_uid}_{image_mode}`                       | Derivable from uid + mode      |


```
lesion_id = {midas_record_id}_{midas_location}_{midas_iscontrol}
                    │
         rank by min(uid) + image_mode
                    │
                    ▼
                 case_id
```

