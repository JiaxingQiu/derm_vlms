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
        "created_at",
        "updated_at",
    )
    readonly_fields = ("created_at", "updated_at")
    ordering = ("case_id", "model")


def _format_duration(seconds):
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    if seconds < 0:
        return ""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


@admin.register(Annotation)
class AnnotationAdmin(admin.ModelAdmin):
    list_display = (
        "dermatologist",
        "case_id",
        "model",
        "short_feedback",
        "first_completion_display",
        "updated_at",
    )
    list_filter = ("dermatologist",)
    search_fields = ("dermatologist__login_id", "case_id", "model")
    readonly_fields = ("first_entered_at", "first_completed_at")

    def short_feedback(self, obj):
        slots = [
            (obj.diagnosis_1 or {}, obj.reasoning_1 or []),
            (obj.diagnosis_2 or {}, obj.reasoning_2 or []),
            (obj.diagnosis_3 or {}, obj.reasoning_3 or []),
        ]
        used = [d for d, r in slots if d.get("name") or d.get("label") or r]
        if not used:
            return ""
        reviewed = sum(1 for d in used if d.get("label") in ("correct", "incorrect"))
        return f"{reviewed}/{len(used)} reviewed"
    short_feedback.short_description = "feedback"

    def first_completion_display(self, obj):
        return _format_duration(obj.first_completion_seconds)
    first_completion_display.short_description = "1st completion"


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
        """Export all annotations as a flat CSV.

        Conditional rows are flattened across the top-3 differential, so
        each diagnosis gets four dedicated columns:

            diag_N_name                  AI-provided diagnosis name
            diag_N_label                 'correct' | 'incorrect' | ''
            reasoning_N                  JSON list of
                {original, edited, edits_made}
                with sentence-level [ev k] crops inlined into ``edited`` as
                ``[crop:{...}]`` markers
            diag_N_correct_differential  free-text alternative when label
                == 'incorrect' (with [ev k] crops inlined as [crop:{...}])

        The unconditional (human-only) flow has been retired and persists
        no data, so it contributes no columns.
        """
        response = HttpResponse(content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = 'attachment; filename="annotations_export.csv"'
        response.write("\ufeff")

        writer = csv.writer(response)
        writer.writerow([
            "login_id",
            "case_id",
            "model",
            "raw_response",
            "diag_1_name", "diag_1_label", "reasoning_1", "diag_1_correct_differential",
            "diag_2_name", "diag_2_label", "reasoning_2", "diag_2_correct_differential",
            "diag_3_name", "diag_3_label", "reasoning_3", "diag_3_correct_differential",
            "other_feedback",
            "first_entered_at",
            "first_completed_at",
            "first_completion_seconds",
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
            per_diag_cols = [""] * 12
            slots = [
                (ann.diagnosis_1 or {}, ann.reasoning_1 or []),
                (ann.diagnosis_2 or {}, ann.reasoning_2 or []),
                (ann.diagnosis_3 or {}, ann.reasoning_3 or []),
            ]
            for k, (d, r) in enumerate(slots):
                label = d.get("label", "") or ""
                edits_out = []
                for edit in r or []:
                    original = (edit or {}).get("original", "")
                    edited = (edit or {}).get("edited", original)
                    edits_out.append({
                        "original": original,
                        "edited": inline_crops(edited, (edit or {}).get("crops", [])),
                        "edits_made": (edited or "") != (original or ""),
                    })
                correct_diff = ""
                if label == "incorrect":
                    correct_diff = inline_crops(
                        d.get("correct_differential", ""),
                        d.get("correct_differential_crops", []),
                    )
                base = k * 4
                per_diag_cols[base + 0] = d.get("name", "") or ""
                per_diag_cols[base + 1] = label
                per_diag_cols[base + 2] = (
                    json.dumps(edits_out, ensure_ascii=False) if edits_out else ""
                )
                per_diag_cols[base + 3] = correct_diff

            other_fb_export = inline_tc(ann.other_feedback)

            writer.writerow([
                ann.dermatologist.login_id,
                ann.case_id,
                ann.model,
                ann.raw_response,
                *per_diag_cols,
                other_fb_export,
                ann.first_entered_at.isoformat() if ann.first_entered_at else "",
                ann.first_completed_at.isoformat() if ann.first_completed_at else "",
                ann.first_completion_seconds if ann.first_completion_seconds is not None else "",
                ann.created_at,
                ann.updated_at,
            ])

        return response

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["export_csv_url"] = "export-csv/"
        return super().changelist_view(request, extra_context=extra_context)
