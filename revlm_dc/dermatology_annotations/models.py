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
    dermoscopy_experience = models.CharField(max_length=50, default="", blank=True)
    zip_code = models.CharField(max_length=20, default="", blank=True)
    registered_at = models.DateTimeField(auto_now_add=True)

    # --- Assignment ---
    assignment_slot = models.PositiveIntegerField(null=True, blank=True)

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

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # --- Completeness ---
    marked_complete = models.BooleanField(default=False)

    # --- Timing ---
    # Full visit history: [{"entered_at": iso, "completed_at": iso|null}, ...]
    page_visits = models.JSONField(default=list, blank=True)

    MAX_VISIT_SECONDS = 30 * 60  # 30 min — visits longer than this are capped

    @property
    def total_duration_seconds(self):
        """Total active seconds across all visits (capped per visit)."""
        total = 0.0
        from datetime import datetime
        for v in (self.page_visits or []):
            e = v.get("entered_at")
            c = v.get("completed_at")
            if e and c:
                t0 = datetime.fromisoformat(e)
                t1 = datetime.fromisoformat(c)
                total += min((t1 - t0).total_seconds(), self.MAX_VISIT_SECONDS)
        return total if total > 0 else None

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
        null=True,
        blank=True,
    )
    pcp_user = models.ForeignKey(
        "PCPUser",
        on_delete=models.CASCADE,
        related_name="tab_sessions",
        null=True,
        blank=True,
    )
    token_hash = models.CharField(max_length=64, unique=True)
    expires_at = models.DateTimeField(default=_default_tab_session_expiry, db_index=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(auto_now=True)

    @property
    def user(self):
        """Return whichever user object is set (Dermatologist or PCPUser)."""
        return self.dermatologist or self.pcp_user

    @property
    def user_role(self):
        if self.dermatologist_id:
            return "Dermatologist"
        if self.pcp_user_id:
            return "PCP"
        return None

    class Meta:
        indexes = [
            models.Index(fields=["dermatologist", "expires_at"]),
        ]

    def __str__(self):
        u = self.user
        return f"{u.login_id} tab session" if u else "orphan tab session"


# =========================================================================
# PCP models — parallel set for PCP users
# =========================================================================

class PCPUser(models.Model):
    login_id = models.CharField(max_length=100, unique=True)

    full_name = models.CharField(max_length=200, default="", blank=True)
    occupation = models.CharField(max_length=200, default="", blank=True)
    years_experience = models.PositiveIntegerField(null=True, blank=True)
    institution = models.CharField(max_length=200, default="", blank=True)
    dermoscopy_experience = models.CharField(max_length=50, default="", blank=True)
    zip_code = models.CharField(max_length=20, default="", blank=True)
    registered_at = models.DateTimeField(auto_now_add=True)

    assignment_slot = models.PositiveIntegerField(null=True, blank=True)

    current_case_index = models.PositiveIntegerField(default=0)
    current_model_index = models.PositiveIntegerField(default=0)
    is_done = models.BooleanField(default=False)

    def __str__(self):
        return self.login_id

    class Meta:
        verbose_name = "PCP User"
        verbose_name_plural = "PCP Users"


class PCPAssignment(models.Model):
    evaluator = models.ForeignKey(
        PCPUser,
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


class PCPAnnotation(models.Model):
    pcp_user = models.ForeignKey(
        PCPUser,
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    case_id = models.CharField(max_length=100)
    model = models.CharField(max_length=100, blank=True, default="")

    INTERFACE_UNCONDITIONAL = "unconditional"
    INTERFACE_CONDITIONAL = "conditional"
    INTERFACE_CHOICES = [
        (INTERFACE_UNCONDITIONAL, "Unconditional"),
        (INTERFACE_CONDITIONAL, "Conditional"),
    ]
    interface_type = models.CharField(
        max_length=20, choices=INTERFACE_CHOICES, default=INTERFACE_CONDITIONAL,
    )

    # --- Unconditional fields (user's own assessment, no AI) ---
    unconditional_data = models.JSONField(default=dict, blank=True)

    # --- Conditional fields (AI response evaluation, same as Annotation) ---
    raw_response = models.TextField(blank=True, default="")
    diagnosis_1 = models.JSONField(default=dict, blank=True)
    reasoning_1 = models.JSONField(default=list, blank=True)
    diagnosis_2 = models.JSONField(default=dict, blank=True)
    reasoning_2 = models.JSONField(default=list, blank=True)
    diagnosis_3 = models.JSONField(default=dict, blank=True)
    reasoning_3 = models.JSONField(default=list, blank=True)
    other_feedback = models.JSONField(default=_empty_text_crops, blank=True)
    diagnosis_order = models.JSONField(default=list, blank=True)

    marked_complete = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    page_visits = models.JSONField(default=list, blank=True)

    MAX_VISIT_SECONDS = 30 * 60

    @property
    def total_duration_seconds(self):
        """Total active seconds across all visits (capped per visit)."""
        total = 0.0
        from datetime import datetime
        for v in (self.page_visits or []):
            e = v.get("entered_at")
            c = v.get("completed_at")
            if e and c:
                t0 = datetime.fromisoformat(e)
                t1 = datetime.fromisoformat(c)
                total += min((t1 - t0).total_seconds(), self.MAX_VISIT_SECONDS)
        return total if total > 0 else None

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["pcp_user", "case_id", "model"],
                name="unique_pcp_case_model",
            )
        ]

    def __str__(self):
        label = self.model or "human"
        return f"{self.pcp_user.login_id} - {self.case_id} - {label}"
