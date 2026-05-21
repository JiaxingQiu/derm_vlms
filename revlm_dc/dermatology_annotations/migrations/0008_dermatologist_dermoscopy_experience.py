from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dermatology_annotations', '0007_remove_annotation_benign_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='dermatologist',
            name='dermoscopy_experience',
            field=models.CharField(blank=True, default='', max_length=50),
        ),
    ]
