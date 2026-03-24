from pathlib import Path
import csv
import json

from django.conf import settings
from django.contrib import admin
from django.http import HttpResponse
from django.urls import path
from django.template.response import TemplateResponse

from .models import Dermatologist, Annotation


def load_annotations_data():
    json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


class AnnotationInline(admin.TabularInline):
    model = Annotation
    extra = 0
    fields = (
        "case_id",
        "model_response_correct",
        "textual_feedback",
        "visual_feedback",
        "created_at",
        "updated_at",
    )
    readonly_fields = ("created_at", "updated_at")
    ordering = ("case_id",)


@admin.register(Annotation)
class AnnotationAdmin(admin.ModelAdmin):
    list_display = (
        "dermatologist",
        "case_id",
        "model_response_correct",
        "short_textual_feedback",
        "updated_at",
    )
    list_filter = ("model_response_correct", "dermatologist")
    search_fields = ("dermatologist__login_id", "case_id", "textual_feedback", "visual_feedback")

    def short_textual_feedback(self, obj):
        if not obj.textual_feedback:
            return ""
        return obj.textual_feedback[:60]
    short_textual_feedback.short_description = "textual feedback"


@admin.register(Dermatologist)
class DermatologistAdmin(admin.ModelAdmin):
    list_display = (
        "login_id",
        "current_case_index",
        "is_done",
        "completed_cases_count",
        "total_cases_count",
        "progress_display",
    )
    search_fields = ("login_id",)
    inlines = [AnnotationInline]
    change_list_template = "admin/dermatology_annotations/dermatologist/change_list.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "export-csv/",
                self.admin_site.admin_view(self.export_csv_view),
                name="dermatology_annotations_dermatologist_export_csv",
            ),
        ]
        return custom_urls + urls

    def get_user_case_count(self, obj):
        data = load_annotations_data()
        return len(data.get(obj.login_id, {}))

    def completed_cases_count(self, obj):
        return obj.annotations.count()
    completed_cases_count.short_description = "completed"

    def total_cases_count(self, obj):
        return self.get_user_case_count(obj)
    total_cases_count.short_description = "total"

    def progress_display(self, obj):
        total = self.get_user_case_count(obj)
        completed = obj.annotations.count()
        if total == 0:
            return "0 / 0"
        return f"{completed} / {total}"
    progress_display.short_description = "progress"

    def export_csv_view(self, request):
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="annotations_export.csv"'

        writer = csv.writer(response)
        writer.writerow([
            "login_id",
            "is_done",
            "current_case_index",
            "case_id",
            "model_response_correct",
            "textual_feedback",
            "visual_feedback",
            "created_at",
            "updated_at",
        ])

        annotations = (
            Annotation.objects
            .select_related("dermatologist")
            .all()
            .order_by("dermatologist__login_id", "case_id")
        )

        for annotation in annotations:
            writer.writerow([
                annotation.dermatologist.login_id,
                annotation.dermatologist.is_done,
                annotation.dermatologist.current_case_index,
                annotation.case_id,
                annotation.model_response_correct,
                annotation.textual_feedback or "",
                annotation.visual_feedback or "",
                annotation.created_at,
                annotation.updated_at,
            ])

        return response

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["export_csv_url"] = "export-csv/"
        return super().changelist_view(request, extra_context=extra_context)
    
# Register your models here.
# admin.site.register(Dermatologist)