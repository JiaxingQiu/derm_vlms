"""Shared base for all Anthropic judges on Azure Foundry.

All Claude models use the same endpoint + key (from tokens.py) and the
Anthropic Messages API shape. Subclasses just set ``name`` and
``_deployment_token`` to pick the deployment.
"""

import base64
from io import BytesIO

from ..prompts import SYSTEM_JUDGE, judge_prompt
from .base import Judge, parse_judge_json


def _image_to_base64(image, max_side=2048, quality=90):
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ClaudeBaseJudge(Judge):
    name = "claude_base"
    _deployment_token = None  # subclass sets this

    def load(self):
        import tokens
        from anthropic import AnthropicFoundry

        self.client = AnthropicFoundry(
            api_key=tokens.AZURE_ANTHROPIC_API_KEY,
            base_url=tokens.AZURE_ANTHROPIC_ENDPOINT,
        )
        self.deployment = getattr(tokens, self._deployment_token)

    def judge(self, image, dx, differential="top_1", max_tokens=512):
        b64 = _image_to_base64(image)
        message = self.client.messages.create(
            model=self.deployment,
            system=SYSTEM_JUDGE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": judge_prompt(dx, differential)},
                    ],
                },
            ],
            max_tokens=max_tokens,
        )
        return parse_judge_json(message.content[0].text, differential)
