import re
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from openai import AzureOpenAI

AZURE_ENDPOINT = "https://sa-qd-mlsj68kc-eastus2.cognitiveservices.azure.com/"
API_VERSION = "2024-12-01-preview"
DEPLOYMENT = "gpt-5.3-chat"


def init_client(api_key):
    """Create and return an AzureOpenAI client."""
    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
    )


def _image_to_data_url(image, max_side=2048, quality=90):
    """Convert a PIL Image to a base64 data URL suitable for the API.

    Resizes if either dimension exceeds max_side to stay within token limits.
    """
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def predict_image(
    client,
    image,
    prompt="Is the lesion malignant or benign, or other?",
    max_tokens=1024,
):
    """Run inference on a single PIL image via Azure GPT-5.3.

    Returns the generated text response.
    """
    data_url = _image_to_data_url(image)

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert board-certified dermatologist. "
                    "Answer concisely and precisely using standard "
                    "dermatological terminology."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_completion_tokens=max_tokens,
    )

    return response.choices[0].message.content


def parse_label(response, labels=("malignant", "benign", "other")):
    """Extract a classification label from free-text response via keyword matching.

    Returns the first label found (case-insensitive), or None.
    """
    text_lower = response.lower()
    for label in labels:
        if re.search(rf"\b{label}\b", text_lower):
            return label
    return None
