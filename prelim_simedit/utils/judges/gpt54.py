"""GPT-5.4 judge (Azure OpenAI — separate resource from GPT-5.3)."""

from openai import AzureOpenAI

from ..prompts import SYSTEM_JUDGE, judge_prompt
from .base import Judge, parse_judge_json

# Reuse the image encoder from gpt53 utils
from ..paths import PROJECT_ROOT, load_module_from_path

_UTILS = PROJECT_ROOT / "collect_ai_response" / "gpt53" / "utils.py"


class GPT54Judge(Judge):
    name = "gpt54"

    def load(self):
        from tokens import (
            AZURE_GPT54_API_KEY,
            AZURE_GPT54_DEPLOYMENT,
            AZURE_GPT54_ENDPOINT,
        )

        self._m = load_module_from_path("gpt53_utils_shared", _UTILS)
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_GPT54_ENDPOINT,
            api_key=AZURE_GPT54_API_KEY,
        )
        self.deployment = AZURE_GPT54_DEPLOYMENT

    def judge(self, image, dx, max_tokens=512):
        data_url = self._m._image_to_data_url(image)
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": SYSTEM_JUDGE},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": data_url, "detail": "high"}},
                        {"type": "text", "text": judge_prompt(dx)},
                    ],
                },
            ],
            max_completion_tokens=max_tokens,
        )
        return parse_judge_json(resp.choices[0].message.content)
