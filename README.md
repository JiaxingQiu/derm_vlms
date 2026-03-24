# Dermatology VLMs

Benchmarking dermatology vision-language models on the MIDAS dataset for skin lesion classification (malignant / benign / other).

# Deployment

Download the sample `data` and `images` folders and store them inside `revlm_dc`. Then, follow the next steps

1. Run `pip install django` in a virtual environment
2. Move to the revlm_dc dir and run `python manage.py makemigrations`
3. Run `python manage.py migrate`
4. Run `python manage.py runserver`

# Management

You can track the progress of each participant by going to the address given by the terminal plus `/admin/name_of_the_app`, e.g. `http://127.0.0.1:8000/admin/dermatology_annotations/`

