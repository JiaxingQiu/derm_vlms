# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

## Store annotations data

In case you need to store the annotations, run the following command:

```python
python upload_to_blob.py configs/blob_config.yaml
```

The blob scripts use the container SAS URL and SAS token directly for the `model-annotations` container. `blob_prefix` will be the name of the folder within the blob where all the contents of your target folder will be stored.
For example, if you want to upload a folder `test_folder` to the container with the following structure

```
  test_folder/
  ├── folder_1/                                
  ├── folder_2/
```

You'd have to use the path to `test_folder` for `source_dir` and use the name `test_folder` as your `blob_prefix`. Else, all subfolders will be stored in the container root by default

```yaml
azure:
  sas_url: "https://YOUR_ACCOUNT_NAME.blob.core.windows.net/model-annotations"
  sas_token: "YOUR_SAS_TOKEN"

upload:
  source_dir: "/absolute/path/to/local/source"
  container_name: "model-annotations"
  blob_prefix: "datasets/revlm_dc"
  overwrite: false
```

Downloading is similar. You first specify the `blob_prefix` of the folder you uploaded, e.g. `test_folder`, and then the `target_dir` where you'll store the contents that reside within `test_folder`. If you don't specify the `test_folder` folder name at the end of your `target_dir`, all the contents will be stored in the local root by default.

If your `sas_url` already includes the query string, you can leave `sas_token` as `null`.

## Annotation interface (Django)

**Get the `results/` folder** (by request) and place it under the project root:
```
  derm_vl
  ms/results/
  ├── images/                                # lesion images
  ├── dermato_llama_predictions_all.csv
  ├── medgemma_predictions_all.csv
  └── gpt53_predictions_all.csv
```

You can download the data from the `model-annotations` container using:

```python
python download_from_blob.py configs/blob_config.yaml
```

The script recreates the subfolder structure locally using the blob names under the configured prefix:

```yaml
download:
  container_name: "model-annotations"
  blob_prefix: "datasets/revlm_dc"
  target_dir: "/absolute/path/to/local/download"
  overwrite: false
```

**First-time setup:** database, parse predictions, RCT assignments, then the dev server.

```bash
cd revlm_dc
python manage.py makemigrations
python manage.py migrate
python manage.py parsedata
python manage.py generate_assignments configs/test_config.yaml
python manage.py runserver 0.0.0.0:8000
```

The config parameters from annotations has the following content (update accordingly)

```yaml
users:
  - user_a
  - user_b
  - user_c
  - user_d
  - user_e
  - user_f
  - user_g
  - user_h
  - user_i
  - user_j
  - user_k
seed: 42
max_lesions: 3
enable_factors:
  - image_mode
```

**After new predictions arrive:** re-parse, regenerate assignments, then run the server.

```bash
cd revlm_dc
python manage.py parsedata
python manage.py generate_assignments configs/test_config.yaml
python manage.py runserver
```

**Just run the interface**, no new results or user, run the server.

```bash
cd revlm_dc
python manage.py runserver
```

`parsedata` reads `results/*_predictions_all.csv`, parses VLM responses into diagnoses + descriptions, writes `revlm_dc/data/annotations_data.json`, and copies combined images to `revlm_dc/images/`. `generate_assignments` builds per-user lesion queues; adjust `--users`, `--max-lesions`, and `--enable-factors` as needed.


### Run admin 

```bash
cd revlm_dc
conda activate dermato_llama
python manage.py createsuperuser
```
Then create user name, email and password. http://localhost:8000/admin/

## Interface Engineering (local)

Generate a self-contained HTML interface for expert review of VLM outputs.

1. **Get the `results/` folder** (by request) and place it under the project root:
  ```
   derm_vlms/results/
   ├── images/                                # Sampled lesion images
   ├── dermato_llama_predictions_all.csv
   ├── medgemma_predictions_all.csv
   └── gpt53_predictions_all.csv
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


| Folder                                | Model             | Base                                 | Params | Link                                                                                                    |
| ------------------------------------- | ----------------- | ------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------- |
| `collect_ai_response/skingpt/`        | SkinGPT-4         | BLIP-2 + LLaMA-2-13B-Chat            | ~14B   | [JoshuaChou2018/SkinGPT-4](https://github.com/JoshuaChou2018/SkinGPT-4)                                 |
| `collect_ai_response/dermato_llama/`  | DermatoLlama      | Llama-3.2-11B-Vision-Instruct + LoRA | ~11B   | [DermaVLM/DermatoLLama-full](https://huggingface.co/DermaVLM/DermatoLLama-full)                         |
| `collect_ai_response/llava_derm/`     | LLaVA-Dermatology | LLaVA-1.5-7B                         | ~7B    | [Esperanto/llava-dermatology-7b-v1.5-hf](https://huggingface.co/Esperanto/llava-dermatology-7b-v1.5-hf) |
| `collect_ai_response/medgemma/`       | MedGemma          | MedGemma-1.5-4B-IT                   | ~4B    | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)                           |
| `collect_ai_response/gpt53/`          | GPT-5.3           | Azure OpenAI (proprietary)           | —      | Azure `gpt-5.3-chat` deployment                                                                         |


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
├── collect_ai_response/   # Model inference notebooks & utilities
│   ├── skingpt/           # SkinGPT-4
│   ├── dermato_llama/     # DermatoLlama
│   ├── llava_derm/        # LLaVA-Dermatology
│   ├── medgemma/          # MedGemma
│   └── gpt53/             # GPT-5.3 (Azure OpenAI)
```
