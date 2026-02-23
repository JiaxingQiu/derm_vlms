# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

## Project Structure

```
derm_vlms/
├── data/                  # MIDAS images (3,418 JPGs)
├── data_share/            # Shared metadata
│   ├── midas_share.parquet
│   └── midas_share_dictionary.json
├── skingpt/               # SkinGPT-4 (BLIP-2 + LLaMA-2-13B)
├── dermato_llama/         # DermatoLlama (Llama-3.2-11B-Vision + LoRA)
└── llava_derm/            # LLaVA-Dermatology (LLaVA-1.5-7B fine-tuned on SCIN)
```

## Models

| Folder | Model | Base | Params | HF Link |
|--------|-------|------|--------|---------|
| `skingpt/` | SkinGPT-4 | BLIP-2 + LLaMA-2-13B-Chat | ~14B | [JoshuaChou2018/SkinGPT-4](https://github.com/JoshuaChou2018/SkinGPT-4) |
| `dermato_llama/` | DermatoLlama | Llama-3.2-11B-Vision-Instruct + LoRA | ~11B | [DermaVLM/DermatoLLama-full](https://huggingface.co/DermaVLM/DermatoLLama-full) |
| `llava_derm/` | LLaVA-Dermatology | LLaVA-1.5-7B | ~7B | [Esperanto/llava-dermatology-7b-v1.5-hf](https://huggingface.co/Esperanto/llava-dermatology-7b-v1.5-hf) |

## Evaluation Protocol

Each model follows the same evaluation:

1. Load the shared dataset (`data_share/midas_share.parquet`, 3,357 rows)
2. Stratified sample: 5 images per class (seed=42) → 15 images total
3. Two prompts per image:
   - **Describe**: "Describe the lesion in detail."
   - **Classify**: "Is the lesion malignant or benign, or other?"
4. Parse predicted label from classification response
5. Save results to `<model>/results/`

## Per-Model Setup

Each model folder contains its own:
- `README.md` — setup instructions and model details
- `requirements.txt` — Python dependencies
- `utils.py` — model loading and inference functions
- `notebooks/<name>_predict.ipynb` — prediction notebook
- `results/` — output CSVs

See individual READMEs for conda environment setup and usage.

## Data

The MIDAS dataset contains 3,357 dermatology images with three-class labels:
- **malignant** (1,391)
- **benign** (1,322)
- **other** (644)
