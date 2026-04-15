from django.db import models


class Dermatologist(models.Model):
    login_id = models.CharField(max_length=100, unique=True)
    current_case_index = models.PositiveIntegerField(default=0)
    current_model_index = models.PositiveIntegerField(default=0)
    is_done = models.BooleanField(default=False)

    def __str__(self):
        return self.login_id


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
    # [{text, label, feedback, crops: [{x,y,w,h},...]}]
    diagnosis_feedback = models.JSONField(default=list, blank=True)
    description_feedback = models.JSONField(default=list, blank=True)
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
