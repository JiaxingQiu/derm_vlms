from pathlib import Path
import csv
import json

from django.conf import settings
from django.contrib import admin
from django.http import HttpResponse
from django.urls import path

from .models import Dermatologist, Annotation
from .views import (
    build_page_sequence,
    find_first_incomplete_page,
    get_case_interface_map,
    get_user_case_ids,
    load_annotations_data as load_annotations_data_view,
    load_users_config,
)


def load_annotations_data():
    json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_total_case_count():
    data = load_annotations_data()
    return len(data)


class AnnotationInline(admin.TabularInline):
    model = Annotation
    extra = 0
    fields = (
        "case_id",
        "model_response_correct",
        "difficulty",
        "textual_feedback",
        "visual_feedback",
        "review_data",
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
        "difficulty",
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
        "current_model_index",
        "is_done",
        "completed_cases_count",
        "total_cases_count",
        "progress_display",
    )
    search_fields = ("login_id",)
    inlines = [AnnotationInline]
    change_list_template = "admin/dermatology_annotations/dermatologist/change_list.html"

    def _get_user_page_sequence(self, obj):
        users_config = load_users_config()
        annotations_data = load_annotations_data_view()
        login_id = obj.login_id

        user_case_ids = get_user_case_ids(users_config, login_id)
        if user_case_ids:
            case_ids = [case_id for case_id in user_case_ids if case_id in annotations_data]
        else:
            case_ids = sorted(annotations_data.keys())

        interface_map = get_case_interface_map(users_config, login_id)
        return build_page_sequence(case_ids, annotations_data, interface_map), annotations_data

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
        pages, _ = self._get_user_page_sequence(obj)
        return len(pages)

    def completed_cases_count(self, obj):
        pages, annotations_data = self._get_user_page_sequence(obj)
        if not pages:
            return 0
        return find_first_incomplete_page(pages, annotations_data, obj)
    completed_cases_count.short_description = "completed"

    def total_cases_count(self, obj):
        return self.get_user_case_count(obj)
    total_cases_count.short_description = "total"

    def progress_display(self, obj):
        total = self.get_user_case_count(obj)
        completed = self.completed_cases_count(obj)
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
            "difficulty",
            "textual_feedback",
            "visual_feedback",
            "review_data",
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
                annotation.difficulty,
                annotation.textual_feedback or "",
                annotation.visual_feedback or "",
                json.dumps(annotation.review_data, ensure_ascii=False),
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
