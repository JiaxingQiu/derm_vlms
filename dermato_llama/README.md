# DermatoLlama

[DermaVLM/DermatoLlama-200k](https://huggingface.co/DermaVLM/DermatoLlama-200k) — LoRA fine-tune of Llama-3.2-11B-Vision-Instruct for dermatology, from the SCALEMED framework.

## Setup

### 1. Create conda environment

```bash
conda create -n dermato_llama python=3.11 -y
conda activate dermato_llama
```

### 2. Install dependencies

```bash
cd your_path_to/derm_vlms/dermato_llama
pip install -r requirements.txt
```

For CUDA 12.6 (optional):
```bash
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

### 3. Register Jupyter kernel

```bash
python -m ipykernel install --user --name dermato_llama --display-name "dermato_llama"
```

### 4. HuggingFace access

The base model `meta-llama/Llama-3.2-11B-Vision-Instruct` is gated. You need to:
1. Accept Meta's license at https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
2. Ensure your HF token (in `../token`) has access

## Usage

Open `notebooks/dermato_llama_predict.ipynb` and run all cells with the `dermato_llama` kernel.

## Model details

- **Base**: meta-llama/Llama-3.2-11B-Vision-Instruct
- **Adapter**: DermaVLM/DermatoLLama-full (LoRA)
- **Output**: Free-text dermatology analysis

## Citation

```bibtex
@article {Yilmaz2025-DermatoLlama-VLM,
    author = {Yilmaz, Abdurrahim and Yuceyalcin, Furkan and Varol, Rahmetullah and Gokyayla, Ece and Erdem, Ozan and Choi, Donghee and Demircali, Ali Anil and Gencoglan, Gulsum and Posma, Joram M. and Temelkuran, Burak},
    title = {Resource-efficient medical vision language model for dermatology via a synthetic data generation framework},
    year = {2025},
    doi = {10.1101/2025.05.17.25327785},
    url = {https://www.medrxiv.org/content/early/2025/07/30/2025.05.17.25327785},
    journal = {medRxiv}
}
```
