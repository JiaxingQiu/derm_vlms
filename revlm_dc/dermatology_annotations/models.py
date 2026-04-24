from datetime import timedelta

from django.db import models
from django.utils import timezone


class Dermatologist(models.Model):
    login_id = models.CharField(max_length=100, unique=True)
    current_case_index = models.PositiveIntegerField(default=0)
    current_model_index = models.PositiveIntegerField(default=0)
    is_done = models.BooleanField(default=False)

    def __str__(self):
        return self.login_id


def _default_tab_session_expiry():
    return timezone.now() + timedelta(hours=12)


def _empty_text_crops():
    return {"text": "", "crops": []}


class Annotation(models.Model):
    dermatologist = models.ForeignKey(
        Dermatologist,
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    case_id = models.CharField(max_length=100)
    model = models.CharField(max_length=100, blank=True, default="")
    interface_type = models.CharField(max_length=20, default="conditional")

    # --- Conditional: AI response evaluation ---
    raw_response = models.TextField(blank=True, default="")
    # Per-diagnosis review entries. Each item:
    #   {
    #     "name": str,                          # AI diagnosis name
    #     "label": "" | "correct" | "incorrect",  # "" or "correct" = accepted
    #     "reasoning_edits": [                   # aligned 1:1 with AI sentences
    #         {"original": str, "edited": str}
    #     ],
    #     "correct_differential": str           # required when label == "incorrect"
    #   }
    diagnosis_feedback = models.JSONField(default=list, blank=True)
    # {text, crops: [{x,y,w,h},...]}
    other_feedback = models.JSONField(default=_empty_text_crops, blank=True)

    # --- Unconditional: human independent assessment ---
    # Each is {text, crops: [{x,y,w,h},...]}
    user_diagnosis_1 = models.JSONField(default=_empty_text_crops, blank=True)
    user_diagnosis_2 = models.JSONField(default=_empty_text_crops, blank=True)
    user_diagnosis_3 = models.JSONField(default=_empty_text_crops, blank=True)
    user_reasons = models.JSONField(default=_empty_text_crops, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["dermatologist", "case_id", "model"],
                name="unique_dermatologist_case_model",
            )
        ]

    def __str__(self):
        label = self.model or "human"
        return f"{self.dermatologist.login_id} - {self.case_id} - {label}"


class TabAuthSession(models.Model):
    dermatologist = models.ForeignKey(
        Dermatologist,
        on_delete=models.CASCADE,
        related_name="tab_sessions",
    )
    token_hash = models.CharField(max_length=64, unique=True)
    expires_at = models.DateTimeField(default=_default_tab_session_expiry, db_index=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["dermatologist", "expires_at"]),
        ]

    def __str__(self):
        return f"{self.dermatologist.login_id} tab session"
