"""DermatoLlama (Llama-3.2-11B-Vision + LoRA) robot.

Wraps the existing collect_ai_response loader. Registered as a candidate
medrobot; uses the same uniform predict() interface as MedGemma.
"""

from ..paths import PROJECT_ROOT, load_module_from_path
from .base import Robot

_UTILS = PROJECT_ROOT / "collect_ai_response" / "dermato_llama" / "utils.py"


class DermatoLlamaRobot(Robot):
    name = "dermato_llama"

    def load(self):
        import torch  # noqa: F401  (lazy; only needed on a GPU node)

        try:
            from tokens import HF_TOKEN
        except Exception:
            HF_TOKEN = None

        self._m = load_module_from_path("dermato_llama_utils", _UTILS)
        self.model, self.processor = self._m.load_model(
            hf_token=HF_TOKEN, **self.kwargs
        )

    def predict(self, image, prompt, max_new_tokens=64):
        # Greedy for short top-1 outputs (stable, reproducible).
        return self._m.predict_image(
            self.model, self.processor, image,
            prompt=prompt, max_new_tokens=max_new_tokens, do_sample=False,
        )
