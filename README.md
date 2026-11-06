# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

# 1. Pipeline


| Step | Folder                                                  | What it does                                             | Input                                     | Output                                                               |
| ---- | ------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| 1    | `[data_utils/](data_utils/README.md)`                   | Process raw MIDAS → shared parquet + case mapping        | `data/release_midas.xlsx` + images        | `data_share/midas_share.parquet` + `data_share/case_mapping.parquet` |
| 2    | `[collect_ai_response/](collect_ai_response/README.md)` | Run VLM inference + visual grounding on all lesions      | `data_share/midas_share.parquet` + images | `results/*_predictions_reason.csv` + `results/images/`               |
| 2.5  | `[prelim_acc/](prelim_acc/README.md)`                   | Automated accuracy check (top-1/top-3 vs ground truth)   | prediction CSVs                           | accuracy tables                                                      |
| 3    | `[revlm_dc/](revlm_dc/README.md)`                       | Annotation interface (Django) — development + deployment | prediction CSVs + images                  | human annotations (PostgreSQL)                                       |


# 2. Infrastructure

- **Azure Blob Storage** — syncs prediction CSVs, images, and deployment data. ([tutorial](https://www.youtube.com/watch?v=sEImMaovc1Q))
- **Django web app** (`revlm_dc/`) — annotation interface on Azure VM. ([tutorial](https://www.youtube.com/watch?v=nGIg40xs9e4&t=103s))
- **PostgreSQL** — production DB for annotations. SQLite for local dev. ([tutorial](https://www.youtube.com/watch?v=HEV1PWycOuQ))

# 3. New Collaborator Setup

Follow these steps to set up the project on a new machine. You will need the repo URL and blob storage credentials (shared offline by an existing team member).

**3.1 Clone the repo and install dependencies**

```bash
git clone -b registration https://github.com/JiaxingQiu/derm_vlms.git
cd derm_vlms
unset PYTHONPATH
conda create -n dermato_llama python=3.11 -y
conda activate dermato_llama
pip install -r requirements_local.txt
```

**3.2 Create `configs/blob_config.yaml`**

This file is gitignored — each collaborator maintains their own copy with local paths. Create it using the template below and fill in the SAS credentials you received:

```yaml
azure:
  sas_url: "https://dermsac.blob.core.windows.net/model-annotations?<SAS_QUERY_STRING>"
  sas_token: "<SAS_QUERY_STRING>"

project_root: "/your/local/path/to/derm_vlms"

uploads:
  - name: "raw source data"
    source_dir: "results/"
    container_name: "model-annotations"
    blob_prefix: "datasets/revlm_dc"
    overwrite: false
  - name: "deployment data"
    source_dir: "revlm_dc/data/"
    container_name: "model-annotations"
    blob_prefix: "deploy/data"
    overwrite: false
  - name: "deployment images"
    source_dir: "revlm_dc/images/"
    container_name: "model-annotations"
    blob_prefix: "deploy/images"
    overwrite: false
  - name: "shared data (parquet + case mapping)"
    source_dir: "data_share/"
    container_name: "model-annotations"
    blob_prefix: "data_share"
    overwrite: false

downloads:
  - name: "deployment data"
    container_name: "model-annotations"
    blob_prefix: "deploy/data"
    target_dir: "revlm_dc/data/"
    overwrite: false
  - name: "deployment images"
    container_name: "model-annotations"
    blob_prefix: "deploy/images"
    target_dir: "revlm_dc/images/"
    overwrite: false
  - name: "shared data (parquet + case mapping)"
    container_name: "model-annotations"
    blob_prefix: "data_share"
    target_dir: "data_share/"
    overwrite: false
```

Set `project_root` to your local clone path. All `source_dir` and `target_dir` paths are relative to it.

**3.3 Download data from blob**

```bash
python download_from_blob.py configs/blob_config.yaml
```

This downloads `data_share/` (parquet + case mapping), `results/` (prediction CSVs + images), and deployment data.

**3.4  Run the annotation interface locally**

Only needed if working on or testing the interface:

```bash
conda activate dermato_llama
export DJANGO_SECRET_KEY="any-random-string-for-local-dev" # first time only
cd revlm_dc
python manage.py migrate       # first time only
python manage.py runserver
```


# 4. Debug mode

If a reviewer reports a bug in the live annotation interface, stop the app on the Azure VM before debugging. Stopping freezes new edits. Annotations already saved in PostgreSQL stay as they are.

Leave Nginx, PostgreSQL, and the VM running. Do not `systemctl disable` the unit, and do not kill Gunicorn by hand. The unit is `Restart=always`, so a manual kill starts it again within a few seconds. The unit stays `enabled`, so a reboot of the VM would start the app again. Leave the VM up while you debug.

Do not run `migrate` or `parsedata` on the server while it is down for a bug.

**4.1 Stop the service**

SSH to `derm-dc-vm` as `azureuser`:

```bash
sudo systemctl stop revlm_dc
sudo systemctl status revlm_dc --no-pager
```

`status` should show `Active: inactive (dead)`, and the process should have exited with `status=0/SUCCESS`.

http://20.246.91.185 returns **502 Bad Gateway** until the service is started again.

**4.2 Debug locally**

Reproduce and fix on your own machine with the local server in section 3.4. Deploy the fix with the steps in [revlm_dc/README.md](revlm_dc/README.md) before turning production back on. `systemctl start` brings back whatever code is already on the VM.

**4.3 Turn the service back on**

On the VM, after the fix is on the server:

```bash
sudo systemctl start revlm_dc
sudo systemctl status revlm_dc --no-pager
curl -I http://127.0.0.1:8000
```

`status` should say `active (running)`. `curl` should return HTTP 200 or 302.

