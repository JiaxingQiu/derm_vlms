"""GPT-5.3 judge (Azure OpenAI).

Reuses the existing collect_ai_response/gpt53 client + image encoding so the
endpoint / deployment / api-version stay in one place.
"""

from ..paths import PROJECT_ROOT, load_module_from_path
from ..prompts import SYSTEM_JUDGE, judge_prompt
from .base import Judge, parse_judge_json

_UTILS = PROJECT_ROOT / "collect_ai_response" / "gpt53" / "utils.py"


class GPT53Judge(Judge):
    name = "gpt53"

    def load(self):
        from tokens import AZURE_GPT53_API_KEY

        self._m = load_module_from_path("gpt53_utils", _UTILS)
        self.client = self._m.init_client(api_key=AZURE_GPT53_API_KEY)
        self.deployment = self._m.DEPLOYMENT

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
