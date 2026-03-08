import re
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(
    model_name="google/medgemma-1.5-4b-it",
    hf_token=None,
    torch_dtype=torch.bfloat16,
    device_map="auto",
):
    """Load MedGemma 1.5 4B IT model. Returns (model, processor)."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

    model.eval()

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def predict_image(
    model,
    processor,
    image,
    prompt="Is the lesion malignant or benign, or other?",
    max_new_tokens=512,
):
    """Run inference on a single PIL image. Returns the generated text.

    Uses greedy decoding as recommended for MedGemma 1.5.
    """
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
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[0][input_length:]
    return processor.decode(generated_tokens, skip_special_tokens=True)


def parse_label(response, labels=("malignant", "benign", "other")):
    """Extract a classification label from free-text response via keyword matching.

    Returns the first label found (case-insensitive), or None.
    """
    text_lower = response.lower()
    for label in labels:
        if re.search(rf"\b{label}\b", text_lower):
            return label
    return None

