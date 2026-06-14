"""Judge ABC: the pluggable judge (stronger reviewer) side.

A judge sees the image + the robot's proposed diagnosis and returns its own
diagnosis. Each judge implements its own client call.
"""

import json
import re
from abc import ABC, abstractmethod


def parse_judge_json(text, differential="top_1"):
    """Robustly extract the judge JSON object from a model response."""
    if not text:
        if differential == "top_1":
            return {"diagnosis": "", "reasoning": ""}
        return {"corrected_differential": ""}

    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        if differential == "top_1":
            return {"diagnosis": "", "reasoning": ""}
        return {"corrected_differential": ""}

    if differential == "top_1":
        return {
            "diagnosis": str(obj.get("diagnosis",
                             obj.get("correct_diagnosis", ""))).strip(),
            "reasoning": str(obj.get("reasoning", "")).strip(),
        }
    else:
        return {
            "corrected_differential": str(
                obj.get("corrected_differential", "")).strip(),
        }


class Judge(ABC):
    name = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._loaded = False

    @abstractmethod
    def load(self):
        """Initialise the API client. Call once."""

    @abstractmethod
    def judge(self, image, dx, differential="top_1"):
        """Return dict with diagnosis fields."""

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True
