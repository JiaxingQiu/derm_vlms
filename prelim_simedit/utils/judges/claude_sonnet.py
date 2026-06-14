"""Claude Sonnet 4.6 judge (Anthropic on Azure Foundry)."""

from .claude_base import ClaudeBaseJudge


class ClaudeSonnet46Judge(ClaudeBaseJudge):
    name = "claude_sonnet46"
    _deployment_token = "AZURE_CLAUDE_SONNET46_DEPLOYMENT"
