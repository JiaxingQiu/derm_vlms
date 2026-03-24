from django.db import models

# Models here
class Dermatologist(models.Model):
    login_id = models.CharField(max_length=100, unique=True)
    current_case_index = models.PositiveIntegerField(default=0)
    is_done = models.BooleanField(default=False)

    def __str__(self):
        return self.login_id

class Annotation(models.Model):
    dermatologist = models.ForeignKey(
        Dermatologist,
        on_delete=models.CASCADE,
        related_name="annotations"
    )
    case_id = models.CharField(max_length=100)

    model_response_correct = models.BooleanField(default=False)
    textual_feedback = models.TextField(blank=True, null=True)
    visual_feedback = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["dermatologist", "case_id"],
                name="unique_dermatologist_case"
            )
        ]

    def __str__(self):
        return f"{self.dermatologist.login_id} - {self.case_id}"