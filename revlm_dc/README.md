# revlm_dc

Django annotation interface for human evaluation of VLM predictions.

> **Warning:** Only the data lead (Jiaxing Qiu) should run `parsedata` or `upload_to_blob.py`. These modify shared production data.

---

## 1. Data Lead Only: Local Setup & Data Pipeline

1. **First-time env setup:**

```bash
conda create -n dermato_llama python=3.11 -y
conda activate dermato_llama
pip install -r requirements_local.txt
```

1. **Data setup** (run once, or when prediction CSVs change):

```bash
conda activate dermato_llama
cd revlm_dc
python manage.py parsedata    # parses CSVs → annotations_data.json + assignment_slots.json + images
cd ..
python upload_to_blob.py configs/blob_config.yaml      # uploads data + images to Azure blob
```

1. **Regular startup** (no data changes):

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

- **Assignments** are pre-computed as 500 static slots by `parsedata` and never recomputed on the server. Once a user registers, their lesion list is immutable. See [ASSIGNMENT.md](readme/ASSIGNMENT.md).
- **Test account:** Log in with username `test` — no registration needed. Resets on each login. Always uses slot 0.
- **Admin panel:** Run `python manage.py createsuperuser` once, then visit [http://localhost:8000/admin/](http://localhost:8000/admin/).
- **Locking a finished user (read-only):** Add their `login_id` to `revlm_dc/data/locked_users.json` (a JSON list, e.g. `["joyyy"]`). They can still log in and browse every case, but nothing they do is saved — no edits, no progress cursor, no timing. Takes effect on their next request; remove the id to unlock. Because `data/` is gitignored, create/edit this file directly wherever the app runs (locally or on the server) — no restart or deploy needed.

---

## 2. Data Lead Only: Re-deploying a New Version

**Before pushing**, make sure you've completed:

- `makemigrations` if `models.py` changed
- `parsedata` + `upload_to_blob.py` if prediction data changed
- Verify the interface runs locally with no errors

Then commit and push:

```bash
git add .
git commit -m "description of changes"
git push origin <your-working-branch-name>
```

> **Important:** Never gitignore the `migrations/` folder. Migration files must be committed from dev so the server only applies them.
>
> **Important:** Never run `parsedata` or `generate_assignments` on the server. All data artifacts are produced locally and uploaded via blob. See [ASSIGNMENT.md](readme/ASSIGNMENT.md).

**On the Azure server:**

**1. Pull the latest code**

```bash
cd /home/azureuser/derm_vlms
git pull origin <your-working-branch-name>
conda activate derm_django_env
```

**(2.) Download deployment artifacts from blob**

```bash
python download_from_blob.py configs/blob_config.yaml
```

This pulls `annotations_data.json`, `assignment_slots.json`, and images into `revlm_dc/data/` and `revlm_dc/images/`.

**(3.) Apply DB migrations**

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

**(7.) Verify**

Visit [http://20.246.91.185](http://20.246.91.185) and test the interface. If something goes wrong:

```bash
sudo journalctl -u revlm_dc -f
sudo tail -f /var/log/nginx/error.log
```

---

**(8.) Admin** [http://20.246.91.185/admin](http://20.246.91.185/admin).

- User: admin
- Pass: gyfxog-manri5-Juvniq

## Documentation

- [ASSIGNMENT.md](readme/ASSIGNMENT.md) — How lesions are assigned to annotators (slot system, registration flow, safety guarantees)
- [DEPLOYMENT.md](readme/DEPLOYMENT.md) — Full server deployment guide (Gunicorn + Nginx + HTTPS + Azure PostgreSQL)



