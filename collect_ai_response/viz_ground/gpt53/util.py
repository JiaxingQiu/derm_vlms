"""Self-grounding utilities for GPT-5.3.

Provides the same interface as viz_ground/qwen3_8b/util.py but uses
GPT-5.3 (Azure OpenAI) to draw bounding boxes for its OWN reasoning
sentences — so the box and reasoning are coherent from a single model.
"""

import json
import os
import re
import sys
from typing import Optional

from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PARSE_DIR = os.path.join(_PROJECT_ROOT, "revlm_dc", "dermatology_annotations")
if _PARSE_DIR not in sys.path:
    sys.path.insert(0, _PARSE_DIR)

_GPT53_DIR = os.path.join(_PROJECT_ROOT, "collect_ai_response", "gpt53")
if _GPT53_DIR not in sys.path:
    sys.path.insert(0, _GPT53_DIR)

from parse import parse_reason_response  # noqa: E402
from utils import init_client, _image_to_data_url, AZURE_ENDPOINT, API_VERSION, DEPLOYMENT  # noqa: E402


GROUNDING_PROMPT_TEMPLATE = (
    "You are examining a dermatological image. You previously provided the "
    "following reasoning sentence about this image:\n\n"
    "\"{sentence}\"\n\n"
    "Please locate the region of the image that this reasoning sentence "
    "refers to. Output ONLY a JSON object with a single key \"bbox_2d\" "
    "whose value is [x1, y1, x2, y2] in 0-1000 coordinates.\n"
    "Example: {{\"bbox_2d\": [120, 200, 450, 600]}}"
)


# ---------------------------------------------------------------------------
# Reasoning sentence parsing
# ---------------------------------------------------------------------------

def parse_reasoning_sentences(reason_text: str) -> list[dict]:
    """Parse reasoning into a flat list of grounding targets.

    Uses ``parse_reason_response`` from ``revlm_dc/.../parse.py`` so targets
    are identical to what annotators see in the Django interface.
    """
    if not reason_text or not isinstance(reason_text, str):
        return []

    diagnoses = parse_reason_response(reason_text)
    targets: list[dict] = []
    for diag in diagnoses:
        name = diag.get("name", "")
        targets.append({"type": "diagnosis", "diagnosis": name, "text": name})
        for sent in diag.get("reasoning_sentences", []):
            targets.append({"type": "sentence", "diagnosis": name, "text": sent})
    return targets


# ---------------------------------------------------------------------------
# Visual grounding inference
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3


def predict_grounding_box(
    client,
    image: Image.Image,
    sentence: str,
    max_tokens: int = 128,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[dict | None, str]:
    """Ask GPT-5.3 to locate the image region for *sentence*.

    Retries up to max_attempts times if the response fails to parse.
    Returns (box_dict_or_None, raw_output_str).
    """
    data_url = _image_to_data_url(image)
    prompt = GROUNDING_PROMPT_TEMPLATE.format(sentence=sentence)

    messages = [
        {
            "role": "system",
            "content": "You are an expert dermatologist. Output ONLY valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    raw_output = ""
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            if "content_policy_violation" in str(e) or "content_filter" in str(e):
                return None, f"[CONTENT_FILTERED] {e}"
            raise
        raw_output = response.choices[0].message.content

        box = _parse_bbox_response(raw_output)
        if box is not None:
            return box, raw_output

    return None, raw_output


def _parse_bbox_response(text: str) -> dict | None:
    """Extract [x1,y1,x2,y2] from model output and convert to {x,y,w,h}."""
    try:
        obj = json.loads(text.strip())
        coords = obj.get("bbox_2d")
        if coords and len(coords) == 4:
            return _coords_to_xywh(coords)
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if match:
        coords = [int(match.group(i)) for i in range(1, 5)]
        return _coords_to_xywh(coords)

    return None


FULL_IMAGE_THRESHOLD = 0.9
MARGIN = 0.10
MIN_BOX_AREA = 0.01
MIN_BOX_SIDE = 0.05


def _coords_to_xywh(coords: list[int]) -> dict:
    """Convert [x1,y1,x2,y2] in 0-1000 space to {x,y,w,h} in 0-1 space.

    Post-processing:
    - If box covers >90% of both dims, shrink inward by 10% margins.
    - If box is near-zero (area <1%), expand to ~10% side while keeping same x,y.
    """
    x1, y1, x2, y2 = [c / 1000.0 for c in coords]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    w = x2 - x1
    h = y2 - y1

    if w > FULL_IMAGE_THRESHOLD and h > FULL_IMAGE_THRESHOLD:
        x1 = x1 + MARGIN
        y1 = y1 + MARGIN
        w = w - 2 * MARGIN
        h = h - 2 * MARGIN
    elif w * h < MIN_BOX_AREA:
        w = MIN_BOX_SIDE
        h = MIN_BOX_SIDE

    return {
        "x": round(x1, 4),
        "y": round(y1, 4),
        "w": round(w, 4),
        "h": round(h, 4),
    }


# ---------------------------------------------------------------------------
# Batch grounding for one row
# ---------------------------------------------------------------------------

def ground_reasoning_sentences(
    client,
    image: Image.Image,
    reason_text: str,
) -> list[dict]:
    """Parse reasoning text and get a grounding box for each target.

    Returns list of dicts:
        [{"type": ..., "diagnosis": ..., "text": ..., "box": {...}|None,
          "raw_output": str}, ...]
    """
    parsed = parse_reasoning_sentences(reason_text)
    results = []
    for entry in parsed:
        box, raw = predict_grounding_box(client, image, entry["text"])
        results.append({
            "type": entry["type"],
            "diagnosis": entry["diagnosis"],
            "text": entry["text"],
            "box": box,
            "raw_output": raw,
        })
    return results


def grounding_results_to_json(results: list[dict]) -> str:
    """Serialise grounding results list to a compact JSON string.

    Drops raw_output to keep CSV cells compact.
    """
    clean = [
        {"type": r["type"], "diagnosis": r["diagnosis"],
         "text": r["text"], "box": r["box"]}
        for r in results
    ]
    return json.dumps(clean, separators=(",", ":"))
