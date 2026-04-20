import re
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from peft import PeftModel


def load_model(
    base_model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
    adapter_path="DermaVLM/DermatoLLama-full",
    hf_token=None,
    torch_dtype=torch.bfloat16,
    device_map="auto",
):
    """Load base VLM and apply LoRA adapter. Returns (model, processor)."""
    model = MllamaForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(base_model_name, token=hf_token)

    model = PeftModel.from_pretrained(model, adapter_path, token=hf_token)
    model.eval()

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total params:     {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def _build_inputs(processor, image, messages):
    """Build model inputs from a message list and image."""
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    return inputs


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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = _build_inputs(processor, image, messages)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=(
                processor.tokenizer.pad_token_id
                if processor.tokenizer.pad_token_id is not None
                else processor.tokenizer.eos_token_id
            ),
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
