"""MedGemma 1.5 4B-IT robot (wraps the existing collect_ai_response loader)."""

from ..paths import PROJECT_ROOT, load_module_from_path
from .base import Robot

_UTILS = PROJECT_ROOT / "collect_ai_response" / "medgemma" / "utils.py"


class MedGemmaRobot(Robot):
    name = "medgemma"

    def load(self):
        import torch  # noqa: F401  (lazy; only needed on a GPU node)

        try:
            from tokens import HF_TOKEN
        except Exception:
            HF_TOKEN = None

        self._m = load_module_from_path("medgemma_utils", _UTILS)
        self.model, self.processor = self._m.load_model(
            hf_token=HF_TOKEN, **self.kwargs
        )

    def predict(self, image, prompt, max_new_tokens=64):
        return self._m.predict_image(
            self.model, self.processor, image,
            prompt=prompt, max_new_tokens=max_new_tokens,
        )
