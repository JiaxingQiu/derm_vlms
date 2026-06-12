# revlm_dc

Django annotation interface for human evaluation of VLM predictions.

---

| # | Command | What it does |
|---|---------|-------------|
| 1 | `python manage.py parsedata` | (one time unless prediction change) Parse prediction CSVs → `annotations_data.json` + `assignment_slots.json` + images |
| 2 | `python upload_to_blob.py` | (one time unless prediction change) Upload data + images to Azure Blob |
| 3 | `python manage.py makemigrations dermatology_annotations` | Generate DB migration files |
| 4 | `python manage.py migrate` | Apply migrations (SQLite locally, PostgreSQL in production) |
| 5 | `python manage.py runserver` | Start local dev server |

---

## Documentation

- [ASSIGNMENT.md](readme/ASSIGNMENT.md) — How lesions are assigned to annotators (slot system, registration flow, safety guarantees)
- [DEPLOYMENT.md](readme/DEPLOYMENT.md) — Full server deployment guide (Gunicorn + Nginx + HTTPS + Azure PostgreSQL)
