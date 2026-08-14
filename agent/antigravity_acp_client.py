"""OpenAI-compatible shim that forwards Hermes requests to the Antigravity CLI ACP.

Mirrors :class:`agent.acp_client_base.BaseACPClient` for the Google
Antigravity CLI (``agy``) ACP mode (JSON-RPC over stdio). Process reuse and
per-prompt ``session/new`` follow the shared ACP client lifecycle (see parent
module).

The actual binary is ``agy``. As of agy 1.0.16 the native ``--acp`` stdio mode
is not yet shipped (see google-antigravity/antigravity-cli#31); the default
argv therefore uses the conventional ``agy --acp`` shape expected by that
feature. Until then, users can point HERMES_ANTIGRAVITY_ACP_COMMAND at a
community adapter such as ``agy-acp`` or ``antigravity-acp``.

Duck-typed env overrides:
  HERMES_ANTIGRAVITY_ACP_COMMAND / ANTIGRAVITY_CLI_PATH  -> binary path
  HERMES_ANTIGRAVITY_ACP_ARGS                            -> argv override
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Any

from agent.acp_client_base import BaseACPClient

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://antigravity"


def _resolve_command() -> str:
    env = (
        os.getenv("HERMES_ANTIGRAVITY_ACP_COMMAND", "").strip()
        or os.getenv("ANTIGRAVITY_CLI_PATH", "").strip()
    )
    if env:
        return env

    try:
        from hermes_cli.auth import _resolve_external_process_command_path

        resolved = _resolve_external_process_command_path("antigravity-acp", "agy")
        if resolved:
            return resolved
    except Exception:
        pass

    return "agy"


def _resolve_args(command: str | None = None) -> list[str]:
    raw = os.getenv("HERMES_ANTIGRAVITY_ACP_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    # Native ACP entrypoint once google-antigravity/antigravity-cli#31 lands.
    # Adapter binaries (agy-acp, antigravity-acp, etc.) are already in ACP
    # server mode and should not receive the native ``--acp`` launch flag.
    if command:
        name = Path(command).name.casefold()
        if name not in {"agy", "agy.exe", "antigravity", "antigravity.exe"}:
            return []
    return ["--acp"]


class AntigravityACPClient(BaseACPClient):
    """Minimal OpenAI-client-compatible facade for Google Antigravity CLI ACP."""

    _acp_display_name = "Antigravity ACP"
    _default_model_name = "antigravity-acp"
    _install_hint = (
        "Install Google Antigravity CLI (agy) or an ACP adapter (agy-acp), "
        "or set HERMES_ANTIGRAVITY_ACP_COMMAND/ANTIGRAVITY_CLI_PATH."
    )
    _acp_marker_base_url = "acp://antigravity"

    _resolve_command = staticmethod(_resolve_command)
    _resolve_args = staticmethod(_resolve_args)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            acp_command=acp_command,
            acp_args=acp_args,
            acp_cwd=acp_cwd,
            command=command,
            args=args,
            **kwargs,
        )

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return super()._create_chat_completion(
            model=model or "antigravity-acp",
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )
