# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

# 1. Pipeline


| Step | Folder                                                  | What it does                                             | Input                                     | Output                                                               |
| ---- | ------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| 1    | [`data_utils/`](data_utils/README.md)                   | Process raw MIDAS → shared parquet + case mapping        | `data/release_midas.xlsx` + images        | `data_share/midas_share.parquet` + `data_share/case_mapping.parquet` |
| 2    | [`collect_ai_response/`](collect_ai_response/README.md) | Run VLM inference + visual grounding on all lesions      | `data_share/midas_share.parquet` + images | `results/*_predictions_reason.csv` + `results/images/`               |
| 2.5  | [`prelim_acc/`](prelim_acc/README.md)                   | Automated accuracy check (top-1/top-3 vs ground truth)   | prediction CSVs                           | accuracy tables                                                      |
| 3    | [`revlm_dc/`](revlm_dc/README.md)                      | Annotation interface (Django) — development + deployment | prediction CSVs + images                  | human annotations (PostgreSQL)                                       |


# 2. Infrastructure

- **Azure Blob Storage** — syncs prediction CSVs, images, and deployment data. ([tutorial](https://www.youtube.com/watch?v=sEImMaovc1Q))
- **Django web app** (`revlm_dc/`) — annotation interface on Azure VM. ([tutorial](https://www.youtube.com/watch?v=nGIg40xs9e4&t=103s))
- **PostgreSQL** — production DB for annotations. SQLite for local dev. ([tutorial](https://www.youtube.com/watch?v=HEV1PWycOuQ))

### Blob Storage (Upload / Download)

```bash
python upload_to_blob.py configs/blob_config.yaml
python download_from_blob.py configs/blob_config.yaml
```

Config: `configs/blob_config.yaml` (gitignored — see [New Collaborator Setup](#new-collaborator-setup) for template).

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

**0. First-time** create conda env if not exist

```bash
conda create -n dermato_llama python=3.11 -y
conda activate dermato_llama
pip install -r requirements_local.txt
```

**0. First-time** data setup run once, or when prediction CSVs change

```bash
conda activate dermato_llama
cd revlm_dc
python manage.py parsedata    # parses CSVs → annotations_data.json + assignment_slots.json + images
cd ..
python upload_to_blob.py      # uploads data + images to Azure blob
```

**1. Regular startup** (no data changes):

```bash
conda activate dermato_llama
cd revlm_dc
python manage.py makemigrations dermatology_annotations
python manage.py migrate
python manage.py runserver
```


| Command          | When to run                                  | What it does                                                            |
| ---------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| `parsedata`      | Once locally, or when prediction CSVs change | Parse CSVs → `annotations_data.json` + `assignment_slots.json` + images |
| `makemigrations` | After changing `models.py`                   | Generate migration files                                                |
| `migrate`        | After `makemigrations`, or on fresh DB       | Apply migrations (SQLite locally, PostgreSQL in production)             |
| `runserver`      | Every time                                   | Start the Django dev server                                             |


### Notes

- **Assignments** are pre-computed as 500 static slots by `parsedata` and never recomputed on the server. Once a user registers, their lesion list is immutable. See `revlm_dc/ASSIGNMENT.md`.
- **Test account:** Log in with username `test` — no registration needed. Resets on each login. Always uses slot 0.
- **Admin panel:** Run `python manage.py createsuperuser` once, then visit [http://localhost:8000/admin/](http://localhost:8000/admin/).

# 5. Re-deploying a New Version of the Interface

**Before pushing**, make sure you've completed the relevant steps in section 4:

- `makemigrations` if `models.py` changed
- `parsedata` + `upload_to_blob.py` if prediction data changed
- Verify the interface runs locally with no errors

Then commit and push:

```bash
git add .
git commit -m "description of changes"
git push origin <branch-name>
```

> **Important:** Never gitignore the `migrations/` folder. Migration files must be committed from dev so the server only applies them.
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

**(4.) Collect static files** (if templates, JS, or CSS changed)

```bash
python manage.py collectstatic --noinput
```

**5. Restart the app service**

```bash
sudo systemctl restart revlm_dc
sudo systemctl status revlm_dc --no-pager
```

**(6.) Reload Nginx** (only if the Nginx config changed)

```bash
sudo nginx -t && sudo systemctl reload nginx
```

**7. Verify**

Visit [http://20.246.91.185](http://20.246.91.185) and test the interface. If something goes wrong:

```bash
sudo journalctl -u revlm_dc -f
sudo tail -f /var/log/nginx/error.log
```

