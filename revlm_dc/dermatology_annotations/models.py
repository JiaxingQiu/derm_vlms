from datetime import timedelta

from django.db import models
from django.utils import timezone


class Dermatologist(models.Model):
    # --- Identity (stored lowercase, case-insensitive) ---
    login_id = models.CharField(max_length=100, unique=True)

    # --- Profile (collected at registration) ---
    full_name = models.CharField(max_length=200, default="", blank=True)
    occupation = models.CharField(max_length=200, default="", blank=True)
    years_experience = models.PositiveIntegerField(null=True, blank=True)
    institution = models.CharField(max_length=200, default="", blank=True)
    registered_at = models.DateTimeField(auto_now_add=True)

    # --- Progress ---
    current_case_index = models.PositiveIntegerField(default=0)
    current_model_index = models.PositiveIntegerField(default=0)
    is_done = models.BooleanField(default=False)

    def __str__(self):
        return self.login_id

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"


class Assignment(models.Model):
    """Which cases an evaluator should annotate, in what order.

    Created either at self-registration (auto-assigned via RCT logic)
    or in bulk via ``python manage.py generate_assignments``.
    """
    evaluator = models.ForeignKey(
        Dermatologist,
        on_delete=models.CASCADE,
        related_name="assignments",
    )
    case_id = models.CharField(max_length=100)
    order = models.PositiveIntegerField()

    class Meta:
        unique_together = [("evaluator", "case_id")]
        ordering = ["evaluator", "order"]

    def __str__(self):
        return f"{self.evaluator.login_id} #{self.order}: {self.case_id}"


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

    # --- Conditional: AI response evaluation ---
    raw_response = models.TextField(blank=True, default="")
    # Six fields, three pairs, one pair per AI differential (top-3).
    #
    # diagnosis_N (dict): verdict + replacement info
    #   {
    #     "name": str,                                  # AI diagnosis name (always preserved)
    #     "label": "" | "correct" | "incorrect",
    #     "correct_differential": str,                  # human replacement diagnosis name
    #   }
    #
    # reasoning_N (list): 1:1 with AI-extracted sentences for diagnosis N
    #   [{"original": str, "edited": str, "crops": [{x,y,w,h}, ...]}, ...]
    #   When user replaces a diagnosis, a sentinel entry is prepended:
    #   [{"original": "deleted", "edited": "<human reasoning>"}, ...AI sentences...]
    #
    # Crops always live next to the text whose [ev N] markers they back, so
    # they never need to be re-aligned across fields.
    diagnosis_1 = models.JSONField(default=dict, blank=True)
    reasoning_1 = models.JSONField(default=list, blank=True)
    diagnosis_2 = models.JSONField(default=dict, blank=True)
    reasoning_2 = models.JSONField(default=list, blank=True)
    diagnosis_3 = models.JSONField(default=dict, blank=True)
    reasoning_3 = models.JSONField(default=list, blank=True)
    # {text, crops: [{x,y,w,h},...]}
    other_feedback = models.JSONField(default=_empty_text_crops, blank=True)
    # User-preferred ordering of the 3 AI diagnoses.
    # [] = accepted AI's original order (equivalent to [0, 1, 2]).
    # [2, 0, 1] = user moved AI's #3 to rank 1, AI's #1 to rank 2, AI's #2 to rank 3.
    diagnosis_order = models.JSONField(default=list, blank=True)

    benign = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # --- Timing (set once, never overwritten) ---
    # First time the user GET-rendered this (case, model) page.
    first_entered_at = models.DateTimeField(null=True, blank=True)
    # First time the user clicked Next/Finish on this page.
    first_completed_at = models.DateTimeField(null=True, blank=True)

    @property
    def first_completion_seconds(self):
        """Wall-clock seconds from first entry to first completion, or None."""
        if not self.first_entered_at or not self.first_completed_at:
            return None
        return (self.first_completed_at - self.first_entered_at).total_seconds()

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
