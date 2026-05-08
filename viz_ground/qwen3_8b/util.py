"""Utilities for visual grounding with Qwen3-VL-8B-Instruct.

Provides functions to:
  - Load the Qwen3-VL model
  - Parse reasoning using the same ``parse_reason_response`` the Django
    interface uses (from ``revlm_dc/dermatology_annotations/parse.py``)
  - Prompt Qwen3-VL to predict a bounding box for each reasoning sentence
    **and** each diagnosis name
  - Convert between Qwen3-VL's [0-1000] coordinate space and normalized [0-1]
"""

import json
import os
import re
import sys
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Re-use the exact parsing logic the Django interface uses so that the
# sentences we ground are identical to what the annotators see.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PARSE_DIR = os.path.join(_PROJECT_ROOT, "revlm_dc", "dermatology_annotations")
if _PARSE_DIR not in sys.path:
    sys.path.insert(0, _PARSE_DIR)
from parse import parse_reason_response  # noqa: E402


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

GROUNDING_PROMPT_TEMPLATE = (
    "You are examining a dermatological image. A dermatologist provided the "
    "following reasoning sentence about this image:\n\n"
    "\"{sentence}\"\n\n"
    "Please locate the region of the image that this reasoning sentence "
    "refers to. Output ONLY a JSON object with a single key \"bbox_2d\" "
    "whose value is [x1, y1, x2, y2] in 0-1000 coordinates.\n"
    "Example: {{\"bbox_2d\": [120, 200, 450, 600]}}"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MAX_IMAGE_SIDE = 1024


def load_model(
    model_id: str = MODEL_ID,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
):
    """Load Qwen3-VL model and processor. Returns (model, processor)."""
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_id}  ({n_params:,} params)")
    return model, processor


def _resize_image(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Down-scale large images to limit GPU memory during vision encoding."""
    if max(image.size) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Reasoning sentence parsing  (delegates to parse.py for consistency)
# ---------------------------------------------------------------------------

def parse_reasoning_sentences(reason_text: str) -> list[dict]:
    """Parse using the same logic as the Django annotation interface.

    Uses ``parse_reason_response`` from ``revlm_dc/.../parse.py`` so the
    diagnosis names and reasoning sentences are *identical* to what the
    annotators see in the browser.

    Returns a flat list of grounding targets:
        [
            {"type": "diagnosis", "diagnosis": str, "text": str},
            {"type": "sentence", "diagnosis": str, "text": str},
            ...
        ]
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

def predict_grounding_box(
    model,
    processor,
    image: Image.Image,
    sentence: str,
    max_new_tokens: int = 128,
) -> dict | None:
    """Ask Qwen3-VL to locate the image region for *sentence*.

    Returns {"x": float, "y": float, "w": float, "h": float} in normalised
    [0, 1] coordinates (matching the frontend grounding_box format), or
    None on failure.
    """
    prompt = GROUNDING_PROMPT_TEMPLATE.format(sentence=sentence)
    image = _resize_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] CUDA OOM during generate, clearing cache")
            torch.cuda.empty_cache()
            return None
        raise
    finally:
        del inputs
        torch.cuda.empty_cache()

    return _parse_bbox_response(generated)


def _parse_bbox_response(text: str) -> dict | None:
    """Extract [x1,y1,x2,y2] from Qwen3-VL output and convert to {x,y,w,h}.

    The model outputs coordinates in 0-1000 space; we normalise to 0-1.
    """
    # Try JSON parse first
    try:
        obj = json.loads(text.strip())
        coords = obj.get("bbox_2d")
        if coords and len(coords) == 4:
            return _coords_to_xywh(coords)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: extract any 4-number list with regex
    match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if match:
        coords = [int(match.group(i)) for i in range(1, 5)]
        return _coords_to_xywh(coords)

    return None


def _coords_to_xywh(coords: list[int]) -> dict:
    """Convert [x1,y1,x2,y2] in 0-1000 space to {x,y,w,h} in 0-1 space."""
    x1, y1, x2, y2 = [c / 1000.0 for c in coords]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return {
        "x": round(x1, 4),
        "y": round(y1, 4),
        "w": round(x2 - x1, 4),
        "h": round(y2 - y1, 4),
    }


# ---------------------------------------------------------------------------
# Batch grounding for one row
# ---------------------------------------------------------------------------

def ground_reasoning_sentences(
    model,
    processor,
    image: Image.Image,
    reason_text: str,
) -> list[dict]:
    """Parse reasoning text and get a grounding box for each target.

    Grounds both diagnosis names and individual reasoning sentences,
    using the same parsing as the Django annotation interface.
    Resizes the image once and reuses it for all targets.

    Returns list of dicts:
        [{"type": "diagnosis"|"sentence", "diagnosis": str,
          "text": str, "box": {x,y,w,h} | None}, ...]
    """
    parsed = parse_reasoning_sentences(reason_text)
    resized = _resize_image(image)
    results = []
    for entry in parsed:
        box = predict_grounding_box(model, processor, resized, entry["text"])
        results.append({
            "type": entry["type"],
            "diagnosis": entry["diagnosis"],
            "text": entry["text"],
            "box": box,
        })
    return results


def grounding_results_to_json(results: list[dict]) -> str:
    """Serialise grounding results list to a compact JSON string."""
    return json.dumps(results, separators=(",", ":"))
