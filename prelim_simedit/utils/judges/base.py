"""Judge ABC: the pluggable judge (stronger reviewer) side.

A judge sees the image + the robot's proposed diagnosis and returns a verdict.
It never receives the ground-truth label. Each judge implements its own client
call, so an Azure OpenAI judge and an Anthropic-on-Foundry judge can coexist
behind the same interface.
"""

import json
import re
from abc import ABC, abstractmethod

_EMPTY = {"verdict": "", "correct_diagnosis": "", "reasoning": ""}


def parse_judge_json(text):
    """Robustly extract the judge JSON object from a model response."""
    if not text:
        return dict(_EMPTY, raw="")
    raw = text
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
        return dict(_EMPTY, raw=raw)
    return {
        "verdict": str(obj.get("verdict", "")).strip().lower(),
        "correct_diagnosis": str(obj.get("correct_diagnosis", "")).strip(),
        "reasoning": str(obj.get("reasoning", "")).strip(),
        "raw": raw,
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
    def judge(self, image, dx):
        """Return dict: verdict, correct_diagnosis, reasoning, raw."""

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True
