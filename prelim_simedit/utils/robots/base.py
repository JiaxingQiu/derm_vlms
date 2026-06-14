"""Robot ABC: the pluggable "medrobot" (local vision model) side.

A robot only needs to know how to load itself and run one image+prompt.
Prompt construction and batch looping live in the stage runners, so adding a
new candidate medrobot is one subclass + one registry entry.
"""

from abc import ABC, abstractmethod


class Robot(ABC):
    name = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._loaded = False

    @abstractmethod
    def load(self):
        """Load weights onto the GPU. Sets internal state; call once."""

    @abstractmethod
    def predict(self, image, prompt, max_new_tokens=64):
        """Run one PIL image + text prompt, return the raw text response."""

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True
