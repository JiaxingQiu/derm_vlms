"""Claude Fable 5 judge (Anthropic on Azure Foundry)."""

from .claude_base import ClaudeBaseJudge


class ClaudeFableJudge(ClaudeBaseJudge):
    name = "claude_fable"
    _deployment_token = "AZURE_CLAUDE_FABLE_DEPLOYMENT"
