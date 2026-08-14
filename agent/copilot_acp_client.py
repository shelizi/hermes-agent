"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
backend. All generic ACP lifecycle lives in :class:`agent.acp_client_base.BaseACPClient`.
"""

from __future__ import annotations

import os
import shlex

from agent.acp_client_base import BaseACPClient


class CopilotACPClient(BaseACPClient):
    """Minimal OpenAI-client-compatible facade for Copilot ACP."""

    _acp_display_name = "Copilot ACP"
    _default_model_name = "copilot-acp"
    _install_hint = (
        "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
    )
    _acp_marker_base_url = "acp://copilot"

    def _resolve_command(self) -> str:
        return (
            os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
            or os.getenv("COPILOT_CLI_PATH", "").strip()
            or "copilot"
        )

    def _resolve_args(self, command: str | None = None) -> list[str]:
        raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
        if not raw:
            return ["--acp", "--stdio"]
        return shlex.split(raw)

    @staticmethod
    def _is_deprecation_message(stderr_text: str) -> bool:
        """True iff stderr looks like the deprecated gh-copilot extension's banner."""
        lower = stderr_text.lower()
        if not any(req in lower for req in ("gh-copilot",)):
            return False
        return any(marker in lower for marker in ("has been deprecated", "no commands will be executed"))
