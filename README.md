# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

# 1. Project Structure

This project has 3 layers:

- **Azure Blob Storage** — stores model prediction CSVs and lesion images (`results/`). Synced via `upload_to_blob.py` / `download_from_blob.py`. ([tutorial](https://www.youtube.com/watch?v=sEImMaovc1Q))
- **Django web app** (`revlm_dc/`) — annotation interface hosted on an Azure VM. ([tutorial](https://www.youtube.com/watch?v=nGIg40xs9e4&t=103s))
- **PostgreSQL** — production database for human annotations, hosted on Azure Database for PostgreSQL. SQLite is used for local development. ([tutorial](https://www.youtube.com/watch?v=HEV1PWycOuQ))

# 2. Data Container

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

### Inference

Each model's `predict_reason.py` (or SLURM job in `jobs/predict_reason/`) runs inference on all MIDAS lesions:

1. Load the shared dataset (`data_share/midas_share.parquet`, 3,357 rows)
2. Prepare all lesions via `prepare_all_lesions()` in `data_utils/utils.py` — for each lesion, creates up to 4 image conditions:
   - **photo** — clinical photo (6in preferred, else 1ft)
   - **dscope** — dermoscopic image
   - **combined** — side-by-side (photo left | dscope right), only when both exist
   - **virtual** — virtual image, when available
3. Ask a differential-diagnosis prompt: *"Give the top 3 diagnoses in your differential, and provide reasoning for each."*
4. Save results to `results/<model>_predictions_reason.csv` with checkpointing (safe to resume)

Output CSV columns:

| Column               | Description                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| `id`                 | Row identifier: `{num}_{mode}` (e.g. `1_photo`, `1_dscope`, `1_combined`) |
| `ground_truth`       | True label (`malignant` / `benign` / `other`)                             |
| `y16`                | Fine-grained diagnosis label (16 classes)                                 |
| `y16_description`    | Human-readable description of `y16`                                       |
| `image_mode`         | Image condition: `photo`, `dscope`, `combined`, or `virtual`              |
| `reason_classify`    | Model response (top-3 differential with reasoning)                        |
| `image_path`         | Path to the prepared lesion image                                         |
| `original_image_name`| Source filename(s) from MIDAS (combined uses `;` separator)               |
| `lesion_id`          | Lesion identifier                                                         |

### Blob Storage (Upload / Download)

Model outputs, deployment data, and images are stored in Azure Blob Storage so collaborators and the server can sync without zipping files. Both scripts read from `configs/blob_config.yaml` (gitignored — see [New Collaborator Setup](#new-collaborator-setup) for the template).

The config supports **multiple upload/download specs** via `uploads` / `downloads` lists:

- **Raw source data** — prediction CSVs and images in `results/`
- **Deployment data** — `annotations_data.json` and `assignment_slots.json` in `revlm_dc/data/`
- **Deployment images** — prepared lesion images in `revlm_dc/images/`

Upload all specs to blob:

```bash
python upload_to_blob.py configs/blob_config.yaml
```

Download all specs from blob:

```bash
python download_from_blob.py configs/blob_config.yaml
```

Set `overwrite: true` per spec to replace existing files. By default, existing files are skipped.

# 3. New Collaborator Setup

Follow these steps to set up the project on a new machine. You will need the repo URL and blob storage credentials (shared offline by an existing team member).

**3.1 Clone the repo and install dependencies**

```bash
git clone <repo-url>
cd derm_vlms
pip install -r requirements.txt
```

**3.2 Create `configs/blob_config.yaml`**

This file is gitignored — each collaborator maintains their own copy with local paths. Create it using the template below and fill in the SAS credentials you received:

```yaml
azure:
  sas_url: "https://dermsac.blob.core.windows.net/model-annotations?<SAS_QUERY_STRING>"
  sas_token: "<SAS_QUERY_STRING>"

uploads:
  - name: "raw source data"
    source_dir: "/your/local/path/to/derm_vlms/results/"
    container_name: "model-annotations"
    blob_prefix: "datasets/revlm_dc"
    overwrite: false
  - name: "deployment data"
    source_dir: "/your/local/path/to/derm_vlms/revlm_dc/data/"
    container_name: "model-annotations"
    blob_prefix: "deploy/data"
    overwrite: true
  - name: "deployment images"
    source_dir: "/your/local/path/to/derm_vlms/revlm_dc/images/"
    container_name: "model-annotations"
    blob_prefix: "deploy/images"
    overwrite: false

downloads:
  - name: "deployment data"
    container_name: "model-annotations"
    blob_prefix: "deploy/data"
    target_dir: "./revlm_dc/data/"
    overwrite: true
  - name: "deployment images"
    container_name: "model-annotations"
    blob_prefix: "deploy/images"
    target_dir: "./revlm_dc/images/"
    overwrite: false
```

Replace the `source_dir` paths with your actual local directories. The `downloads` section uses relative paths so it works on any machine.

**3.3 Download model outputs from blob**

```bash
python download_from_blob.py configs/blob_config.yaml
```

This recreates the `results/` folder with all CSVs and images.

**3.4 Set up Django and run the interface**

Follow the [Local Deployment](#local-deployment) steps below. For local development, SQLite is used by default — no PostgreSQL or `.env.production` file is needed. You only need to set a Django secret key:

```bash
export DJANGO_SECRET_KEY="any-random-string-for-local-dev"
```

**3.5 Making and sharing changes**

- **Code changes (templates, views, models, etc.):** commit and push via git as usual. If you changed `models.py`, generate migrations first (`python manage.py makemigrations dermatology_annotations`) and commit the migration files.
- **New or updated model outputs:** after placing new CSVs/images in `results/`, upload to blob so others can pull them:

```bash
python upload_to_blob.py configs/blob_config.yaml
```

Other collaborators then run `python download_from_blob.py configs/blob_config.yaml` to sync.

# 4. Local Deployment and Run

Create conda env if not exist:

```bash
conda create -n dermato_llama python=3.11 -y
conda activate dermato_llama
pip install -r requirements_local.txt
```

**First-time data setup (SKIP UNLESS JOY TOLD YOU TO RUN)** (run once, or when prediction CSVs change):

```bash
conda activate dermato_llama
cd revlm_dc
python manage.py parsedata    # parses CSVs → annotations_data.json + assignment_slots.json + images
cd ..
python upload_to_blob.py      # uploads data + images to Azure blob
```

**Regular startup** (no data changes):

```bash
conda activate dermato_llama
cd revlm_dc
python manage.py makemigrations dermatology_annotations
python manage.py migrate
python manage.py runserver
```

| Step                   | What it does                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `makemigrations`       | Generate migration files from `models.py` changes                                                                            |
| `migrate`              | Apply migrations to the database (SQLite locally, PostgreSQL in production)                                                  |
| `parsedata`            | Parse prediction CSVs into `data/annotations_data.json`, copy images, and generate `data/assignment_slots.json` (500 slots)  |
| `runserver`            | Start the Django dev server                                                                                                  |

> **Note:** `generate_assignments` is deprecated. Assignments are now handled via the static slot file generated by `parsedata`. See `revlm_dc/ASSIGNMENT.md` for details.


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

- **User management:** Users register through the web interface (login → "New? Register"). On registration, the system collects name, occupation, institution, and assigns lesions from the next available pre-computed slot (see `revlm_dc/ASSIGNMENT.md`).
- **Test account:** Log in with username `test` — no registration required. Each login wipes previous annotations and resets to page 1. Test accounts always use slot 0.
- **Assignment safety:** Once a user registers, their lesion list is immutable. No server-side command can change it. See `revlm_dc/ASSIGNMENT.md` for the full design.

# 5. Re-deploying a New Version of the Interface

**Prerequisites (on your dev machine, before pushing):**

1. Make sure the interface runs locally with no errors.

2. Generate migration files if `models.py` changed:

```bash
cd revlm_dc
conda activate dermato_llama
python manage.py makemigrations dermatology_annotations
```

3. If prediction CSVs or parsing logic changed, re-run the data pipeline and upload:

```bash
cd revlm_dc
python manage.py parsedata    # regenerates annotations_data.json + assignment_slots.json + images
cd ..
python upload_to_blob.py      # uploads all deployment artifacts to blob
```

4. Commit everything including migration files and push:

```bash
git add .
git commit -m "description of changes"
git push origin <branch-name>
```

> **Important:** Never gitignore the `migrations/` folder. Migration files must be committed from dev so the server only applies them — never generates them.
>
> **Important:** Never run `parsedata` or `generate_assignments` on the server. All data artifacts are produced locally and uploaded via blob. See `revlm_dc/ASSIGNMENT.md`.

**On the Azure server:**

**1. Pull the latest code**

```bash
cd /home/azureuser/derm_vlms
git pull origin <branch-name>
conda activate derm_django_env
```

**2. Download deployment artifacts from blob**

```bash
python download_from_blob.py configs/blob_config.yaml
```

This pulls `annotations_data.json`, `assignment_slots.json`, and images into `revlm_dc/data/` and `revlm_dc/images/`.

**3. Apply DB migrations**

```bash
cd revlm_dc
python manage.py showmigrations   # check for unapplied migrations (no [X])
python manage.py migrate          # apply them to PostgreSQL
```

**4. Collect static files** (if templates, JS, or CSS changed)

```bash
python manage.py collectstatic --noinput
```

**5. Restart the app service**

```bash
sudo systemctl restart revlm_dc
sudo systemctl status revlm_dc --no-pager
```

**6. Reload Nginx** (only if the Nginx config changed)

```bash
sudo nginx -t && sudo systemctl reload nginx
```

**7. Verify**

Visit [http://20.246.91.185](http://20.246.91.185) and test the interface. If something goes wrong:

```bash
sudo journalctl -u revlm_dc -f
sudo tail -f /var/log/nginx/error.log
```

