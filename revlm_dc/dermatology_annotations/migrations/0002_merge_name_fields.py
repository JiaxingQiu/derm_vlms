from django.db import migrations, models


def combine_names(apps, schema_editor):
    Dermatologist = apps.get_model("dermatology_annotations", "Dermatologist")
    for d in Dermatologist.objects.all():
        parts = [d.first_name, d.last_name]
        d.full_name = " ".join(p for p in parts if p).strip()
        d.save(update_fields=["full_name"])


class Migration(migrations.Migration):

    dependencies = [
        ("dermatology_annotations", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="dermatologist",
            name="full_name",
            field=models.CharField(blank=True, default="", max_length=200),
        ),
        migrations.RunPython(combine_names, migrations.RunPython.noop),
        migrations.RemoveField(
            model_name="dermatologist",
            name="first_name",
        ),
        migrations.RemoveField(
            model_name="dermatologist",
            name="last_name",
        ),
    ]
