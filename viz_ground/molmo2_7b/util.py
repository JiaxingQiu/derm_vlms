"""Utilities for visual grounding with Molmo2-O-7B.

Provides functions to:
  - Load the Molmo2-O-7B model
  - Parse reasoning using the same ``parse_reason_response`` the Django
    interface uses (from ``revlm_dc/dermatology_annotations/parse.py``)
  - Prompt Molmo2 to predict a bounding box for each reasoning sentence
    **and** each diagnosis name
  - Parse Molmo2's native <points> coordinate format
  - Convert between Molmo2's [0-1000] coordinate space and normalized [0-1]
"""

import json
import os
import re
import sys
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Molmo2 remote code expects "default" in ROPE_INIT_FUNCTIONS.
# Patch it in if missing (standard RoPE without scaling).
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
            import math
            base = config.rope_theta
            dim = int(config.hidden_size // config.num_attention_heads)
            if hasattr(config, "partial_rotary_factor"):
                dim = int(dim * config.partial_rotary_factor)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
except (ImportError, AttributeError):
    pass

# Re-use the exact parsing logic the Django interface uses so that the
# sentences we ground are identical to what the annotators see.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PARSE_DIR = os.path.join(_PROJECT_ROOT, "revlm_dc", "dermatology_annotations")
if _PARSE_DIR not in sys.path:
    sys.path.insert(0, _PARSE_DIR)
from parse import parse_reason_response  # noqa: E402


MODEL_ID = "allenai/Molmo2-O-7B"
MAX_IMAGE_SIDE = 1024

# Molmo2 supports pointing natively, so we ask it to point to the top-left
# and bottom-right corners of the relevant region.
GROUNDING_PROMPT_TEMPLATE = (
    "You are examining a dermatological image. A dermatologist provided the "
    "following reasoning sentence about this image:\n\n"
    "\"{sentence}\"\n\n"
    "Point to the top-left corner and the bottom-right corner of the region "
    "in the image that this reasoning sentence refers to."
)

# Molmo2 outputs: <points coords="idx x y, idx x y"/>  (0-1000 space)
COORD_REGEX = re.compile(r"<(?:points|tracks)[^>]*coords=\"([0-9\t:;, .]+)\"[^>]*/?>")
POINTS_REGEX = re.compile(r"(\d+)\s+(\d{2,4})\s+(\d{2,4})")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = MODEL_ID,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
):
    """Load Molmo2-O-7B model and processor. Returns (model, processor)."""
    kwargs = {"trust_remote_code": True, "dtype": "auto", "device_map": device_map}
    if hf_token:
        kwargs["token"] = hf_token

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
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
    max_new_tokens: int = 256,
) -> dict | None:
    """Ask Molmo2 to locate the image region for *sentence*.

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
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = processor.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] CUDA OOM during generate, clearing cache")
            torch.cuda.empty_cache()
            return None
        raise
    finally:
        del inputs
        torch.cuda.empty_cache()

    return _parse_points_response(generated)


def _parse_points_response(text: str) -> dict | None:
    """Extract points from Molmo2's <points> output and derive a bounding box.

    Molmo2 outputs coordinates in 0-1000 space via <points coords="..."/>.
    If two points are found, they are treated as top-left / bottom-right corners.
    If only one point is found, a default-sized box is centred on it.
    """
    points = _extract_points(text)
    if not points:
        return None

    if len(points) >= 2:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs) / 1000.0, max(xs) / 1000.0
        y1, y2 = min(ys) / 1000.0, max(ys) / 1000.0
        w = max(x2 - x1, 0.05)
        h = max(y2 - y1, 0.05)
        return {
            "x": round(x1, 4),
            "y": round(y1, 4),
            "w": round(w, 4),
            "h": round(h, 4),
        }

    # Single point: create a box centred on it
    cx, cy = points[0][0] / 1000.0, points[0][1] / 1000.0
    w, h = 0.15, 0.15
    x = max(cx - w / 2, 0.0)
    y = max(cy - h / 2, 0.0)
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "w": round(w, 4),
        "h": round(h, 4),
    }


def _extract_points(text: str) -> list[tuple[float, float]]:
    """Pull (x, y) pairs from Molmo2's output text."""
    points = []

    # Try structured <points coords="..."/> format first
    for coord_match in COORD_REGEX.finditer(text):
        for pt in POINTS_REGEX.finditer(coord_match.group(1)):
            x, y = float(pt.group(2)), float(pt.group(3))
            points.append((x, y))

    if points:
        return points

    # Fallback: bare "idx x y" patterns outside tags
    for pt in POINTS_REGEX.finditer(text):
        x, y = float(pt.group(2)), float(pt.group(3))
        points.append((x, y))

    return points


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
