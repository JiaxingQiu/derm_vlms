# Visual Grounding Pipeline

Generates bounding boxes for each reasoning sentence in VLM predictions.

## Two approaches

| Approach | Grounder | Output suffix | Description |
|----------|----------|---------------|-------------|
| **External** | Qwen3-8B | `_viz.csv` | Separate model draws boxes for all VLMs' sentences |
| **Self** | Each VLM | `_viz_self.csv` | Each VLM draws boxes for its own sentences |

Both produce identical CSV schemas with columns: `viz_grounding`, `viz_grounding_dscope`, `viz_grounding_clinical`, `viz_grounding_clinical_remapped`, `viz_grounding_dscope_remapped`.

## External grounding (Qwen3-8B)

```
# 1. Run grounding (one job per VLM's predictions)
python qwen3_8b/run_viz_ground.py

# 2. Remap clinical/dscope boxes to combined-image coordinates
python remap_boxes_to_combined.py
```

Jobs: `jobs/viz_ground/qwen3_8b/*.sbatch`

## Self-grounding (GPT-5.3, MedGemma, DermatoLlama)

```
# 1. Run grounding (includes remapping inline)
python gpt53/run_self_ground.py
python medgemma/run_self_ground.py
python dermato_llama/run_self_ground.py
```

For large runs, shard with `--start`, `--end`, `--shard`:
```
python gpt53/run_self_ground.py --start 0 --end 327 --shard 0
```

Then merge shards into final CSV:
```
python gpt53/run_self_ground.py --merge
python medgemma/run_self_ground.py --merge
```

Jobs: `jobs/viz_ground/self/*.sbatch`

## Shared logic

- **Reasoning parsing**: All scripts use `revlm_dc/dermatology_annotations/parse.py` (`parse_reason_response`) — same parser as the Django interface.
- **Remapping**: Clinical/dscope boxes are remapped to combined-image coordinates using split ratio from original image dimensions.
- **Retry**: Up to 3 attempts per sentence on parse failure (sampling on retries).
- **Resume**: Scripts skip already-completed rows on restart.

## Directory structure

```
viz_ground/
├── README.md
├── remap_boxes_to_combined.py      # post-process for external grounding
├── qwen3_8b/
│   ├── util.py                     # model loading, inference, parsing
│   ├── run_viz_ground.py           # production script
│   └── notebooks/
├── gpt53/
│   ├── util.py
│   ├── run_self_ground.py
│   └── notebooks/
├── medgemma/
│   ├── util.py
│   ├── run_self_ground.py
│   └── notebooks/
└── dermato_llama/
    ├── util.py
    ├── run_self_ground.py
    └── notebooks/
```
