import django.db.models.deletion
from django.db import migrations, models

import dermatology_annotations.models


class Migration(migrations.Migration):

    dependencies = [
        ("dermatology_annotations", "0008_dermatologist_dermoscopy_experience"),
    ]

    operations = [
        # --- PCPUser ---
        migrations.CreateModel(
            name="PCPUser",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("login_id", models.CharField(max_length=100, unique=True)),
                ("full_name", models.CharField(blank=True, default="", max_length=200)),
                ("occupation", models.CharField(blank=True, default="", max_length=200)),
                ("years_experience", models.PositiveIntegerField(blank=True, null=True)),
                ("institution", models.CharField(blank=True, default="", max_length=200)),
                ("dermoscopy_experience", models.CharField(blank=True, default="", max_length=50)),
                ("zip_code", models.CharField(blank=True, default="", max_length=20)),
                ("registered_at", models.DateTimeField(auto_now_add=True)),
                ("current_case_index", models.PositiveIntegerField(default=0)),
                ("current_model_index", models.PositiveIntegerField(default=0)),
                ("is_done", models.BooleanField(default=False)),
            ],
            options={
                "verbose_name": "PCP User",
                "verbose_name_plural": "PCP Users",
            },
        ),
        # --- PCPAssignment ---
        migrations.CreateModel(
            name="PCPAssignment",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("case_id", models.CharField(max_length=100)),
                ("order", models.PositiveIntegerField()),
                (
                    "evaluator",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="assignments",
                        to="dermatology_annotations.pcpuser",
                    ),
                ),
            ],
            options={
                "ordering": ["evaluator", "order"],
                "unique_together": {("evaluator", "case_id")},
            },
        ),
        # --- PCPAnnotation ---
        migrations.CreateModel(
            name="PCPAnnotation",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("case_id", models.CharField(max_length=100)),
                ("model", models.CharField(blank=True, default="", max_length=100)),
                (
                    "interface_type",
                    models.CharField(
                        choices=[("unconditional", "Unconditional"), ("conditional", "Conditional")],
                        default="conditional",
                        max_length=20,
                    ),
                ),
                ("unconditional_data", models.JSONField(blank=True, default=dict)),
                ("raw_response", models.TextField(blank=True, default="")),
                ("diagnosis_1", models.JSONField(blank=True, default=dict)),
                ("reasoning_1", models.JSONField(blank=True, default=list)),
                ("diagnosis_2", models.JSONField(blank=True, default=dict)),
                ("reasoning_2", models.JSONField(blank=True, default=list)),
                ("diagnosis_3", models.JSONField(blank=True, default=dict)),
                ("reasoning_3", models.JSONField(blank=True, default=list)),
                ("other_feedback", models.JSONField(blank=True, default=dermatology_annotations.models._empty_text_crops)),
                ("diagnosis_order", models.JSONField(blank=True, default=list)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("page_visits", models.JSONField(blank=True, default=list)),
                (
                    "pcp_user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="annotations",
                        to="dermatology_annotations.pcpuser",
                    ),
                ),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=["pcp_user", "case_id", "model"],
                        name="unique_pcp_case_model",
                    )
                ],
            },
        ),
        # --- TabAuthSession: make dermatologist nullable, add pcp_user FK ---
        migrations.AlterField(
            model_name="tabauthsession",
            name="dermatologist",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="tab_sessions",
                to="dermatology_annotations.dermatologist",
            ),
        ),
        migrations.AddField(
            model_name="tabauthsession",
            name="pcp_user",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="tab_sessions",
                to="dermatology_annotations.pcpuser",
            ),
        ),
    ]
