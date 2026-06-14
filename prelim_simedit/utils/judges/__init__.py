"""Judge registry. Add a new judge here to make it plug-and-play."""

from .gpt53 import GPT53Judge
from .gpt54 import GPT54Judge
from .claude_opus import ClaudeOpus48Judge, ClaudeOpus46Judge
from .claude_sonnet import ClaudeSonnet46Judge
from .claude_fable import ClaudeFableJudge

JUDGE_REGISTRY = {
    GPT53Judge.name: GPT53Judge,
    GPT54Judge.name: GPT54Judge,
    ClaudeOpus48Judge.name: ClaudeOpus48Judge,
    ClaudeOpus46Judge.name: ClaudeOpus46Judge,
    ClaudeSonnet46Judge.name: ClaudeSonnet46Judge,
    ClaudeFableJudge.name: ClaudeFableJudge,
}


def get_judge(name, **kwargs):
    if name not in JUDGE_REGISTRY:
        raise KeyError(f"Unknown judge '{name}'. Available: {list(JUDGE_REGISTRY)}")
    return JUDGE_REGISTRY[name](**kwargs)
