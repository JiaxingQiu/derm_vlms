# LLaVA-Dermatology

[Esperanto/llava-dermatology-7b-v1.5-hf](https://huggingface.co/Esperanto/llava-dermatology-7b-v1.5-hf) — LLaVA 1.5 7B fine-tuned with LoRA on the Google SCIN dermatology dataset for skin condition diagnosis.

## Setup

### 1. Create conda environment

```bash
conda create -n llava_derm python=3.11 -y
conda activate llava_derm
```

### 2. Install dependencies

```bash
cd /scratch/jq2uw/derm_vlms/llava_derm
pip install -r requirements.txt
```

### 3. Register Jupyter kernel

```bash
python -m ipykernel install --user --name llava_derm --display-name "llava_derm"
```

### 4. Run

Get a GPU node, launch JupyterLab, open `notebooks/llava_derm_predict.ipynb`, select the **llava_derm** kernel, and run all cells.

## Model details

- **Base**: llava-hf/llava-1.5-7b-hf
- **Fine-tuned on**: Google SCIN dataset (10k+ dermatology images, 5k volunteers)
- **Output**: Free-text dermatology diagnosis and description
- **No gated access**: Model weights are publicly available (no HF token needed)

## Reference

- Blog: [Semantic Understanding of Dermatology Images Using LLaVA and RAG](https://www.esperanto.ai/blog/semantic-understanding-of-dermatology-images-using-llava-and-rag/)
- Dataset: [Google SCIN (Skin Condition Image Network)](https://github.com/google-research-datasets/scin)
