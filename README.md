# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

## Interface Engineering (local)

Generate a self-contained HTML interface for expert review of VLM outputs.

1. **Get the `results/` folder** (by request) and place it under the project root:
  ```
   derm_vlms/results/
   ├── images/                                # Sampled lesion images
   ├── dermato_llama_predictions_paired.csv
   ├── medgemma_predictions_paired.csv
   └── gpt53_predictions_paired.csv
  ```
2. **Run the notebook** `res_eng/interface/notebook.ipynb`
3. **Output** is saved to `results/interface_share/`:
  ```
   results/interface_share/
   ├── index.html    # Self-contained review interface
   └── images/       # Combined lesion images used by the interface
  ```

The `interface_share/` folder is ready to zip and hand off for database integration and online deployment.

---

## Data

Available upon request.

## Models


| Folder           | Model             | Base                                 | Params | Link                                                                                                    |
| ---------------- | ----------------- | ------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------- |
| `skingpt/`       | SkinGPT-4         | BLIP-2 + LLaMA-2-13B-Chat            | ~14B   | [JoshuaChou2018/SkinGPT-4](https://github.com/JoshuaChou2018/SkinGPT-4)                                 |
| `dermato_llama/` | DermatoLlama      | Llama-3.2-11B-Vision-Instruct + LoRA | ~11B   | [DermaVLM/DermatoLLama-full](https://huggingface.co/DermaVLM/DermatoLLama-full)                         |
| `llava_derm/`    | LLaVA-Dermatology | LLaVA-1.5-7B                         | ~7B    | [Esperanto/llava-dermatology-7b-v1.5-hf](https://huggingface.co/Esperanto/llava-dermatology-7b-v1.5-hf) |
| `medgemma/`      | MedGemma          | MedGemma-1.5-4B-IT                   | ~4B    | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)                           |
| `gpt53/`         | GPT-5.3           | Azure OpenAI (proprietary)           | —      | Azure `gpt-5.3-chat` deployment                                                                         |


Each model folder contains its own:

- `README.md` — setup instructions and model details
- `requirements.txt` — Python dependencies
- `utils.py` — model loading and inference functions
- `notebooks/<name>_predict.ipynb` — prediction notebook

## Inference

In their notebook, each model does the same inference:

1. Load the shared dataset (`data_share/midas_share.parquet`, 3,357 rows)
2. Sample 10 lesions (5 malignant + 5 benign, seed=42) that each have both a clinical photo (6in preferred, else 1ft) and a dermoscopic image — via `sample_lesions()` in `data_utils/utils.py`
3. For each lesion, evaluate three image conditions:
  - **photo** — clinical photo at 6in or 1ft
  - **dscope** — dermoscopic image only
  - **combined** — side-by-side (photo left | dscope right)
4. For each image condition, ask three prompts:
  - **describe**: "Describe the lesion in detail."
  - **classify**: "Is the lesion malignant or benign, or other?"
  - **describe_then_classify**: both prompts combined
5. Save results to `results/<model>_predictions_paired.csv` (30 rows per model: 10 lesions × 3 image conditions)

csv columns:


| Column                   | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `id`                     | Row identifier: `{num}_{mode}` (e.g. `1_photo`, `1_dscope`, `1_combined`) |
| `ground_truth`           | True label (malignant / benign)                                           |
| `image_mode`             | Image condition: `photo`, `dscope`, or `combined`                         |
| `describe`               | Model response to the describe prompt                                     |
| `classify`               | Model response to the classify prompt                                     |
| `describe_then_classify` | Model response to the combined prompt                                     |
| `original_image_name`    | Source filename(s) from MIDAS (combined uses `;` separator)               |
| `lesion_id`              | Integer lesion identifier                                                 |


## Project Structure

```
derm_vlms/
├── data/                  # MIDAS images (3,418 JPGs)
├── data_share/            # Shared metadata
│   ├── midas_share.parquet
│   └── midas_share_dictionary.json
├── data_utils/            # Data processing pipeline & sampling utilities
├── results/               # Model outputs + interface_share/ (by request)
├── res_eng/               # Interface engineering
│   └── interface/         # HTML interface builder (notebook + assets)
├── skingpt/               # SkinGPT-4
├── dermato_llama/         # DermatoLlama
├── llava_derm/            # LLaVA-Dermatology
├── medgemma/              # MedGemma
└── gpt53/                 # GPT-5.3 (Azure OpenAI)
```

