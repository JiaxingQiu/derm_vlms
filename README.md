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


| Folder                               | Model             | Base                                 | Params | Link                                                                                                    |
| ------------------------------------ | ----------------- | ------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------- |
| `collect_ai_response/skingpt/`       | SkinGPT-4         | BLIP-2 + LLaMA-2-13B-Chat            | ~14B   | [JoshuaChou2018/SkinGPT-4](https://github.com/JoshuaChou2018/SkinGPT-4)                                 |
| `collect_ai_response/dermato_llama/` | DermatoLlama      | Llama-3.2-11B-Vision-Instruct + LoRA | ~11B   | [DermaVLM/DermatoLLama-full](https://huggingface.co/DermaVLM/DermatoLLama-full)                         |
| `collect_ai_response/llava_derm/`    | LLaVA-Dermatology | LLaVA-1.5-7B                         | ~7B    | [Esperanto/llava-dermatology-7b-v1.5-hf](https://huggingface.co/Esperanto/llava-dermatology-7b-v1.5-hf) |
| `collect_ai_response/medgemma/`      | MedGemma          | MedGemma-1.5-4B-IT                   | ~4B    | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)                           |
| `collect_ai_response/gpt53/`         | GPT-5.3           | Azure OpenAI (proprietary)           | —      | Azure `gpt-5.3-chat` deployment                                                                         |


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


# Local Deployment

```bash
conda activate dermato_llama
cd revlm_dc
python manage.py makemigrations dermatology_annotations
python manage.py migrate
python manage.py parsedata
python manage.py generate_assignments
python manage.py runserver
```

| Step | What it does |
|------|-------------|
| `makemigrations` | Generate migration files from `models.py` changes |
| `migrate` | Apply migrations to the database (SQLite locally, PostgreSQL in production) |
| `parsedata` | Parse `results/*_predictions_reason.csv` and `*_viz.csv` into `data/annotations_data.json`, copy images to `images/` |
| `generate_assignments` | Assign lesions to all users: 26 shared IRR + 75 random per user (see `assignments.py`). Supports `--users`, `--dry-run` |
| `runserver` | Start the Django dev server |

### Resetting the local database

**Local dev only — never do this in production.**

```bash
rm -f db.sqlite3
find dermatology_annotations/migrations -type f ! -name '__init__.py' -delete
find dermatology_annotations/migrations -type d -name '__pycache__' -exec rm -rf {} +
```

Then re-run the full setup above.

### Admin panel

```bash
python manage.py createsuperuser
```

Then visit [http://localhost:8000/admin/](http://localhost:8000/admin/). The admin panel shows per-user progress, assignments, and supports CSV export of all annotations.

### Notes

- **User management:** Users register through the web interface (login → "New? Register"). On registration, the system collects name, occupation, institution, and auto-assigns lesions.
- **Test account:** Log in with username `test` — no registration required. Each login wipes previous annotations and resets to page 1.



# Re-deploying a New Version of the Interface

**Prerequisites (on your dev machine, before pushing):**

1. Make sure the interface runs locally with no errors
2. Generate migration files if `models.py` changed:

```bash
cd revlm_dc
conda activate ...
python manage.py makemigrations dermatology_annotations
```

3. If new prediction or visual-grounding CSVs were generated, upload them to the blob:

```bash
cd ..
python upload_to_blob.py configs/blob_config.yaml
```

4. Commit everything including migration files and push:

```bash
git add .
git commit -m "description of changes"
git push origin <branch-name>
```

> **Important:** Never gitignore the `migrations/` folder. Migration files must be committed from dev so the server only applies them — never generates them.

**On the Azure server:**

**1. Pull the latest code**

```bash
cd /home/azureuser/derm_vlms
git pull origin <branch-name>
```

**2. Download new data from blob** (if new CSVs were uploaded)

```bash
python download_from_blob.py configs/blob_config.yaml
```

**3. Apply DB migrations**

```bash
cd revlm_dc
conda activate derm_django_env
python manage.py showmigrations   # check for unapplied migrations (no [X])
python manage.py migrate          # apply them to PostgreSQL
```

**4. Re-parse data** (if prediction CSVs or parsing logic changed)

```bash
python manage.py parsedata
```

**5. Regenerate assignments** (if assignment logic, lesion counts, or user list changed)

```bash
python manage.py generate_assignments
```

This uses defaults from `assignments.py` (`IRR_COUNT=26` shared lesions + `RANDOM_COUNT=75` per-user random lesions = 101 total). Use `--users alice bob` to target specific users, or `--dry-run` to preview without writing.

**6. Collect static files** (if templates, JS, or CSS changed)

```bash
python manage.py collectstatic --noinput
```

**7. Restart the app service**

```bash
sudo systemctl restart revlm_dc
sudo systemctl status revlm_dc --no-pager
```

**8. Reload Nginx** (only if the Nginx config changed)

```bash
sudo nginx -t && sudo systemctl reload nginx
```

**9. Verify**

Visit [http://20.246.91.185](http://20.246.91.185) and test the interface. If something goes wrong:

```bash
sudo journalctl -u revlm_dc -f
sudo tail -f /var/log/nginx/error.log
```

