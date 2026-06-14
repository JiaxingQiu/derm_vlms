"""Self-grounding utilities for MedGemma.

Provides the same interface as viz_ground/qwen3_8b/util.py but uses
MedGemma (google/medgemma-1.5-4b-it) to draw bounding boxes for its OWN
reasoning sentences — so the box and reasoning are coherent from a single model.
"""

import json
import os
import re
import sys
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PARSE_DIR = os.path.join(_PROJECT_ROOT, "revlm_dc", "dermatology_annotations")
if _PARSE_DIR not in sys.path:
    sys.path.insert(0, _PARSE_DIR)

from parse import parse_reason_response  # noqa: E402


MODEL_ID = "google/medgemma-1.5-4b-it"
MAX_IMAGE_SIDE = 1024

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
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = MODEL_ID,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
):
    """Load MedGemma model and processor. Returns (model, processor)."""
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, **kwargs)
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


def preprocess_image_inputs(processor, image: Image.Image, device):
    """Pre-process image once and return reusable pixel_values tensor.

    Call this once per image, then pass the result to predict_grounding_box
    for each sentence to avoid redundant image encoding.
    """
    image = _resize_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "x"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    return inputs.get("pixel_values"), image


def predict_grounding_box(
    model,
    processor,
    image: Image.Image,
    sentence: str,
    max_new_tokens: int = 128,
    max_attempts: int = MAX_ATTEMPTS,
    cached_pixel_values=None,
) -> tuple[dict | None, str]:
    """Ask MedGemma to locate the image region for *sentence*.

    Retries up to max_attempts times if the response fails to parse.
    On retries, enables sampling (temperature=0.4) to get varied output.
    If cached_pixel_values is provided, reuses pre-computed image tensor
    to avoid expensive re-processing per sentence.
    Returns (box_dict_or_None, raw_output_str).
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

    raw_output = ""
    for attempt in range(max_attempts):
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        if cached_pixel_values is not None:
            inputs["pixel_values"] = cached_pixel_values

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(attempt > 0),
                temperature=0.4 if attempt > 0 else 1.0,
            )
        raw_output = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        del inputs

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
    model,
    processor,
    image: Image.Image,
    reason_text: str,
) -> list[dict]:
    """Parse reasoning text and get a grounding box for each target.

    Returns list of dicts:
        [{"type": ..., "diagnosis": ..., "text": ..., "box": {...}|None,
          "raw_output": str}, ...]
    """
    parsed = parse_reasoning_sentences(reason_text)
    resized = _resize_image(image)
    results = []
    for entry in parsed:
        box, raw = predict_grounding_box(model, processor, resized, entry["text"])
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
