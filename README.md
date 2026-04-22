# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

# Project Structure

This project has 3 layers

```
  data_container/
  website/
  psql_server/
```

- **Data Container:** Server designed to store large amounts of (un)structured data. In out setup, we use it for storing the annotations from our VLMs. [This](https://www.youtube.com/watch?v=sEImMaovc1Q) tutorial goes through all the components, mainly Storage Accounts and Blob Containers, which are the ones used for our project.
- **Website:** Developed in Django and hosted in a server (Azure Virtual Machine). Recommended tutorial for Django [here](https://www.youtube.com/watch?v=nGIg40xs9e4&t=103s)
- **PostgreSQL (PSQL):** Database engine hosted on a separate server (Azure Database for PSQL) for storing our data collected from the website. It's handled through Django, so it helps to understand how to connect the Django app to the PSQL server by checking [this](https://www.youtube.com/watch?v=HEV1PWycOuQ) tutorial. For debugging purposes, feel free to use the default Sqlite. However, you need a PostgreSQL for production as Sqlite is not designed for handling production streams of data.

As a suggestion, go through each tutorial to understand the basics of how to deploy our system as everything is connected for deployment.

# Data Container

We use a container to easily manage our model annotation data described below


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

You can obtain the predictions from the models by running their corresponding notebooks:

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

If you are planning to add more annotations while keeping the same folder structure and files, you can use the same instructions as above. 

In case you need to overwrite some files, set the `overwrite` option to `true`.

## Download Annotations Data

Downloading is similar. You first specify the `blob_prefix` of the folder you uploaded, e.g. `test_folder`, and then the `target_dir` where you'll store the contents that reside within `test_folder`. If you don't specify the `test_folder` folder name at the end of your `target_dir`, all the contents will be stored in the local root by default. If your `sas_url` already includes the query string, you can leave `sas_token` as `null`.

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

# Website

## Local Deploymemt

### First-time setup

The following codes will create the database, the migrations (Python scripts that create the columns specified in `models.py`)

```bash
cd revlm_dc
python manage.py makemigrations
python manage.py migrate
python manage.py parsedata
python manage.py generate_assignments configs/test_config.yaml
python manage.py runserver 0.0.0.0:8000
```

`parsedata` reads `results/*_predictions_all.csv`, parses VLM responses into diagnoses + descriptions, writes `revlm_dc/data/annotations_data.json`, and copies combined images to `revlm_dc/images/`. `generate_assignments` builds per-user lesion queues; adjust `--users`, `--max-lesions`, and `--enable-factors` as needed.

The parameters from `configs/test_config.yaml` has the following content (update accordingly)

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

### Deployment after Updates to Database

**WARNING:** Do NOT do it in production or all data will be lost. Do it only during Debug/Local Deployments.

In case there are modifications to `models.py` (fields stored), run the following codes

Locally, we can reset the database 

```bash
rm -f db.sqlite3
find dermatology_annotations/migrations -type f ! -name '__init__.py' -delete
find dermatology_annotations/migrations -type d -name '__pycache__' -exec rm -rf {} +

python manage.py makemigrations dermatology_annotations
python manage.py migrate

python manage.py runserver
```

In production, we can avoid resetting the database by updating any pending migrations

```bash
# if you changed models.py locally
python manage.py makemigrations dermatology_annotations

# apply pending migrations to postgres
python manage.py migrate

# optional checks
python manage.py showmigrations
```

### Updating Server after New Annotations/Predictions Arrive

Re-parse, regenerate assignments, then run the server.

```bash
cd revlm_dc
python manage.py parsedata
python manage.py generate_assignments configs/test_config.yaml
python manage.py runserver
```

### Run admin 

We use an admin tab for supervising data collection in real time.

To enable it, run the following commands

```bash
cd revlm_dc
conda activate dermato_llama
python manage.py createsuperuser
```

Then create user name, email and password. http://localhost:8000/admin/

# PostgreSQL Server

Follow the instructions from this tutorial to deploy the server on [Azure](https://www.youtube.com/watch?v=sEImMaovc1Q)

# Production Deployment

Follow `revml_dc/DEPLOYMENT.md`.

## Pulling Changes to Server

You need to run the following command

```bash
git pull upstream main
```

## Pushing Changes from Server

You need to run the following command

```bash
git add .
git commit -m "name of commit"
git push upstream main
```
