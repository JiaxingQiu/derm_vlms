"""Claude Opus judges (Anthropic on Azure Foundry)."""

from .claude_base import ClaudeBaseJudge


class ClaudeOpus48Judge(ClaudeBaseJudge):
    name = "claude_opus48"
    _deployment_token = "AZURE_CLAUDE_OPUS48_DEPLOYMENT"


class ClaudeOpus46Judge(ClaudeBaseJudge):
    name = "claude_opus46"
    _deployment_token = "AZURE_CLAUDE_OPUS46_DEPLOYMENT"
