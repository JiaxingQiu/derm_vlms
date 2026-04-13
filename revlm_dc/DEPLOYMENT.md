# Deploy revlm_dc Django App (Public Website)

This guide deploys the app with Gunicorn + Nginx + HTTPS (Let's Encrypt) on Ubuntu.

## 1) Install system packages

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

## 2) Install Python server packages in your conda env

```bash
source /home/azureuser/miniconda3/etc/profile.d/conda.sh
conda activate derm_django_env
pip install gunicorn
```

## 3) Configure Django environment (separate env file)

Create a private production env file from the example:

```bash
cd /home/azureuser/derm_vlms/revlm_dc
cp .env.production.example .env.production
chmod 600 .env.production
```

Edit `.env.production` and set at least:
- DJANGO_SECRET_KEY: strong unique value
- DJANGO_ALLOWED_HOSTS: your domain(s), comma-separated
- DJANGO_CSRF_TRUSTED_ORIGINS: include https:// domain URLs

The real `.env.production` file is gitignored.

## 4) Prepare static files and database

```bash
cd /home/azureuser/derm_vlms/revlm_dc
source /home/azureuser/miniconda3/etc/profile.d/conda.sh
conda activate derm_django_env
python manage.py migrate
python manage.py collectstatic --noinput
```

Optional admin user:

```bash
python manage.py createsuperuser
```

## 5) Create and start Gunicorn systemd service

```bash
sudo cp deploy/gunicorn.service.example /etc/systemd/system/revlm_dc.service
sudo systemctl daemon-reload
sudo systemctl enable revlm_dc
sudo systemctl start revlm_dc
sudo systemctl status revlm_dc --no-pager
```

## 6) Configure Nginx

```bash
sudo cp deploy/nginx.revlm_dc.conf.example /etc/nginx/sites-available/revlm_dc
sudo ln -sf /etc/nginx/sites-available/revlm_dc /etc/nginx/sites-enabled/revlm_dc
sudo nginx -t
sudo systemctl restart nginx
```

If default site conflicts:

```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 7) Open firewall (if UFW enabled)

```bash
sudo ufw allow 'Nginx Full'
sudo ufw status
```

## 8) Enable HTTPS certificate

Point your domain DNS A record to this server first, then run:

```bash
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Certbot auto-installs renewal timer. Verify:

```bash
sudo systemctl status certbot.timer --no-pager
```

## 9) Verify deployment

- Visit https://your-domain.com
- Check logs:

```bash
sudo journalctl -u revlm_dc -f
sudo tail -f /var/log/nginx/error.log
```

## Notes

- Django production settings are environment-driven in revlm_dc/settings.py.
- DEBUG defaults to False.
- If you need local dev behavior, set DJANGO_DEBUG=True temporarily.

## Azure PostgreSQL (Make Database Persistent)

Use one of these approaches in `.env.production`:

1) Single URL:

```bash
DATABASE_URL=postgresql://dbuser:dbpassword@your-server.postgres.database.azure.com:5432/your_db_name
DJANGO_DB_SSLMODE=require
```

2) Discrete variables:

```bash
DJANGO_DB_ENGINE=postgresql
DJANGO_DB_NAME=your_db_name
DJANGO_DB_USER=dbuser
DJANGO_DB_PASSWORD=dbpassword
DJANGO_DB_HOST=your-server.postgres.database.azure.com
DJANGO_DB_PORT=5432
DJANGO_DB_SSLMODE=require
```

Azure AD token auth option (recommended when you already use `az`):

```bash
DJANGO_DB_ENGINE=postgresql
DJANGO_DB_NAME=postgres
DJANGO_DB_USER=your-aad-user@tenant.onmicrosoft.com
DJANGO_DB_HOST=your-server.postgres.database.azure.com
DJANGO_DB_PORT=5432
DJANGO_DB_SSLMODE=require
DJANGO_DB_PASSWORD_COMMAND=az account get-access-token --resource https://ossrdbms-aad.database.windows.net --query accessToken --output tsv
```

Important: `.env.production` is not a shell script, so `$(...)` is not executed there.
Use `DJANGO_DB_PASSWORD_COMMAND` instead of `DJANGO_DB_PASSWORD="$(...)"`.

Before using `DJANGO_DB_PASSWORD_COMMAND`, log in on the VM:

```bash
az login --use-device-code
```

Install PostgreSQL driver in your env:

```bash
source /home/azureuser/miniconda3/etc/profile.d/conda.sh
conda activate derm_django_env
pip install "psycopg[binary]"
```

Apply migrations on Azure PostgreSQL:

```bash
cd /home/azureuser/derm_vlms/revlm_dc
python manage.py migrate
```

If you want to move existing SQLite data to Azure PostgreSQL:

```bash
# 1) Export from current SQLite state
DJANGO_DB_ENGINE=sqlite python manage.py dumpdata --natural-foreign --natural-primary -e contenttypes -e auth.permission --indent 2 > db_export.json

# 2) Switch .env.production to Azure PostgreSQL values

# 3) Migrate and load
python manage.py migrate
python manage.py loaddata db_export.json
```

After DB switch, restart app service:

```bash
sudo systemctl restart revlm_dc
```
