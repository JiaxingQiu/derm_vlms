# Collect AI Response

Run VLM inference on all MIDAS lesions and (optionally) compute visual grounding boxes.

---

## Models

| Folder | Model | Base | Params | Link |
|--------|-------|------|--------|------|
| `skingpt/` | SkinGPT-4 | BLIP-2 + LLaMA-2-13B-Chat | ~14B | [JoshuaChou2018/SkinGPT-4](https://github.com/JoshuaChou2018/SkinGPT-4) |
| `dermato_llama/` | DermatoLlama | Llama-3.2-11B-Vision-Instruct + LoRA | ~11B | [DermaVLM/DermatoLLama-full](https://huggingface.co/DermaVLM/DermatoLLama-full) |
| `llava_derm/` | LLaVA-Dermatology | LLaVA-1.5-7B | ~7B | [Esperanto/llava-dermatology-7b-v1.5-hf](https://huggingface.co/Esperanto/llava-dermatology-7b-v1.5-hf) |
| `medgemma/` | MedGemma | MedGemma-1.5-4B-IT | ~4B | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) |
| `gpt53/` | GPT-5.3 | Azure OpenAI (proprietary) | — | Azure `gpt-5.3-chat` deployment |

Each model folder contains: `README.md`, `requirements.txt`, `utils.py`, `predict_reason.py`, `notebooks/`

---

## Inference

| # | Command | Input | Output |
|---|---------|-------|--------|
| 1 | `python collect_ai_response/<model>/predict_reason.py` | `data_share/midas_share.parquet` + `data/images/midas/` | `results/<model>_predictions_reason.csv` + `results/images/` |
| 2 | `python collect_ai_response/viz_ground/<grounding_model>/run_viz_ground.py` | `results/*_predictions_reason.csv` + `results/images/` | `results/*_predictions_reason_viz.csv` |
| 3 | `python collect_ai_response/viz_ground/remap_boxes_to_combined.py` | `results/*_predictions_reason_viz.csv` + images | Updated `*_viz.csv` with remapped box columns |

**Step 1** for each model:
1. Load `midas_share.parquet`
2. Call `prepare_all_lesions()` — creates up to 4 image conditions per lesion (photo, dscope, combined, virtual)
3. Prompt: *"Give the top 3 diagnoses in your differential, and provide reasoning for each."*
4. Save with checkpointing (safe to resume)

---

## Output CSV Columns (`*_predictions_reason.csv`)

| Column | Description |
|--------|-------------|
| `id` | `case_id`: `{case_num}_{image_mode}` (e.g. `1_photo`, `1_combined`) |
| `ground_truth` | True label: `malignant` / `benign` / `other` |
| `y16` | Fine-grained diagnosis (16 classes) |
| `y16_description` | Human-readable description of `y16` |
| `image_mode` | `photo`, `dscope`, `combined`, or `virtual` |
| `reason_classify` | Model response (top-3 differential with reasoning) |
| `image_path` | Path to prepared lesion image |
| `original_image_name` | Source filename(s) from MIDAS (combined uses `;` separator) |
| `lesion_id` | Lesion identifier (maps back to `midas_share.parquet`) |
