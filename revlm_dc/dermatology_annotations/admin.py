import csv
import json
import re
from pathlib import Path

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

_EMPTY_TC = {"text": "", "crops": []}


def load_annotations_data():
    json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_total_case_count():
    data = load_annotations_data()
    return len(data)


def inline_crops(text, crops):
    """Replace [ev N] markers with [crop:{...}] using the Nth crop rect."""
    def _replace(m):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(crops or []):
            return "[crop:" + json.dumps(crops[idx]) + "]"
        return m.group(0)
    return re.sub(r"\[ev\s+(\d+)\]", _replace, text or "")


def inline_tc(tc):
    """Inline crops into a {text, crops} dict, returning the final string."""
    if not tc or not isinstance(tc, dict):
        return ""
    return inline_crops(tc.get("text", ""), tc.get("crops", []))


class AnnotationInline(admin.TabularInline):
    model = Annotation
    extra = 0
    fields = (
        "case_id",
        "model",
        "interface_type",
        "created_at",
        "updated_at",
    )
    readonly_fields = ("created_at", "updated_at")
    ordering = ("case_id", "model")


@admin.register(Annotation)
class AnnotationAdmin(admin.ModelAdmin):
    list_display = (
        "dermatologist",
        "case_id",
        "model",
        "interface_type",
        "short_feedback",
        "updated_at",
    )
    list_filter = ("interface_type", "dermatologist")
    search_fields = ("dermatologist__login_id", "case_id", "model")

    def short_feedback(self, obj):
        if obj.interface_type == "unconditional":
            diags = [
                (obj.user_diagnosis_1 or _EMPTY_TC).get("text", ""),
                (obj.user_diagnosis_2 or _EMPTY_TC).get("text", ""),
                (obj.user_diagnosis_3 or _EMPTY_TC).get("text", ""),
            ]
            filled = [d for d in diags if d.strip()]
            return f"{len(filled)} diagnoses" if filled else ""
        items = obj.diagnosis_feedback or []
        reviewed = sum(1 for i in items if i.get("label") in ("correct", "incorrect"))
        return f"{reviewed}/{len(items)} reviewed" if items else ""
    short_feedback.short_description = "feedback"


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
        """Export all annotations in the unified CSV format.

        Conditional rows export the new reason-based schema:
            diagnosis_feedback JSON = [
                {
                    "name": str,
                    "label": "correct" | "incorrect" | "",
                    "reasoning_edits": [
                        {"original": str, "edited": str, "edits_made": bool}
                    ],
                    "correct_differential": str
                }, ...
            ]
        Crops from ``reasoning_edits[*].crops`` and
        ``correct_differential_crops`` are inlined into the corresponding
        text as ``[crop:{...}]`` markers.
        Unconditional rows keep user_diagnosis_1/2/3 and user_reasons.
        """
        response = HttpResponse(content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = 'attachment; filename="annotations_export.csv"'
        response.write("\ufeff")

        writer = csv.writer(response)
        writer.writerow([
            "login_id",
            "case_id",
            "model",
            "interface_type",
            "raw_response",
            "diagnosis_feedback",
            "other_feedback",
            "user_diagnosis_1",
            "user_diagnosis_2",
            "user_diagnosis_3",
            "user_reasons",
            "created_at",
            "updated_at",
        ])

        annotations = (
            Annotation.objects
            .select_related("dermatologist")
            .all()
            .order_by("dermatologist__login_id", "case_id", "model")
        )

        for ann in annotations:
            diag_fb_export = ""
            other_fb_export = ""
            ud1 = ud2 = ud3 = ur = ""

            if ann.interface_type == "conditional":
                diag_items = []
                for item in ann.diagnosis_feedback or []:
                    label = item.get("label", "")
                    edits_out = []
                    for edit in item.get("reasoning_edits") or []:
                        original = edit.get("original", "")
                        edited = edit.get("edited", original)
                        edits_out.append({
                            "original": original,
                            "edited": inline_crops(edited, edit.get("crops", [])),
                            "edits_made": (edited or "") != (original or ""),
                        })
                    correct_diff = ""
                    if label == "incorrect":
                        correct_diff = inline_crops(
                            item.get("correct_differential", ""),
                            item.get("correct_differential_crops", []),
                        )
                    diag_items.append({
                        "name": item.get("name", ""),
                        "label": label,
                        "reasoning_edits": edits_out,
                        "correct_differential": correct_diff,
                    })

                diag_fb_export = json.dumps(diag_items, ensure_ascii=False) if diag_items else ""
                other_fb_export = inline_tc(ann.other_feedback)
            else:
                ud1 = inline_tc(ann.user_diagnosis_1)
                ud2 = inline_tc(ann.user_diagnosis_2)
                ud3 = inline_tc(ann.user_diagnosis_3)
                ur = inline_tc(ann.user_reasons)

            writer.writerow([
                ann.dermatologist.login_id,
                ann.case_id,
                ann.model,
                ann.interface_type,
                ann.raw_response,
                diag_fb_export,
                other_fb_export,
                ud1,
                ud2,
                ud3,
                ur,
                ann.created_at,
                ann.updated_at,
            ])

        return response

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["export_csv_url"] = "export-csv/"
        return super().changelist_view(request, extra_context=extra_context)
