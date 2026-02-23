import re
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor


def load_model(
    model_name="Esperanto/llava-dermatology-7b-v1.5-hf",
    torch_dtype=torch.float16,
    device_map="auto",
):
    """Load LLaVA-Dermatology model. Returns (model, processor)."""
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Fix for transformers>=4.57: older LLaVA 1.5 checkpoints don't set patch_size
    if processor.patch_size is None:
        processor.patch_size = model.config.vision_config.patch_size
    if processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    # Account for the CLS token so the processor expands <image> to the correct
    # number of placeholder tokens (patches + 1 CLS, minus 1 for "default" = patches).
    processor.num_additional_image_tokens = 1

    model.eval()

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def predict_image(
    model,
    processor,
    image,
    prompt="Is the lesion malignant or benign, or other?",
    max_new_tokens=512,
    temperature=0.4,
    top_p=0.95,
    do_sample=True,
):
    """Run inference on a single PIL image. Returns the generated text."""
    input_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    inputs = processor(
        images=image,
        text=input_text,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
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
