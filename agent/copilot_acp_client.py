"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
backend. All generic ACP lifecycle lives in :class:`agent.acp_client_base.BaseACPClient`.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from typing import Any

from agent.acp_client_base import BaseACPClient, _format_messages_as_prompt

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://copilot"

_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = ("has been deprecated", "no commands will be executed")

# Probe verdicts cached per binary path so repeated prompts against a
# CLI that supports --acp pay the ~50ms --help cost exactly once per
# process. Only definitive verdicts (True/False) are cached; an
# inconclusive probe (binary missing, --help crashed or timed out) is
# not cached so a CLI installed mid-session is picked up.
_ACP_PROBE_CACHE: dict[str, bool] = {}


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""
    lower = stderr_text.lower()
    if not any(req in lower for req in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lower for marker in _DEPRECATION_MARKERS)


def _acp_supported(command: str, args: list[str]) -> bool | None:
    """Tri-state probe: does ``command`` accept the ACP args we would pass?

    Different CLI versions support different transports. The GitHub
    Copilot CLI (``@github/copilot``, late 2025+) ships with ``--acp``;
    older releases (and Claude Code v2.x as of Aug 2026) do not.
    Spawning a CLI that does not recognize the flag silently exits
    with code 1 and ``error: unknown option '--acp'`` on stderr,
    after which every delegate_task call hangs the parent for
    ``child_timeout_seconds`` (default 600s) waiting for stdout
    that never arrives.

    Returns:
      - ``True``  — help text advertises ``--acp``; safe to spawn.
      - ``False`` — help ran cleanly but ``--acp`` is absent; spawning
        would hang, so the caller should fast-fail with a clear error.
      - ``None``  — inconclusive (binary missing, --help failed or
        timed out). The caller must fall through to the normal spawn
        path, which surfaces the existing "Could not start Copilot ACP
        command" error with full context.
    """
    # Only probes when ``--acp`` is actually among ``args``: a custom
    # HERMES_COPILOT_ACP_ARGS transport is the operator's business.
    if "--acp" not in args:
        return True
    cached = _ACP_PROBE_CACHE.get(command)
    if cached is not None:
        return cached
    try:
        probe = subprocess.run(
            [command, "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if probe.returncode != 0:
        # --help itself failed; can't tell anything about --acp.
        return None
    stdout = probe.stdout or ""
    # Match ``--acp`` as a flag in the help text; tolerate spacing and
    # variants like ``[--acp]``.
    verdict = bool(re.search(r"(?:^|[\s\[])--acp(?:[\s=\],]|$)", stdout, re.MULTILINE))
    _ACP_PROBE_CACHE[command] = verdict
    return verdict


def _enabled_ids(entries: Any, key: str) -> set[str]:
    """Ids of ``entries`` (dicts) whose ``_meta.copilotEnablement`` is not ``disabled``."""
    return {
        str(e.get(key) or "").strip()
        for e in (entries or [])
        if isinstance(e, dict)
        and str((e.get("_meta") or {}).get("copilotEnablement") or "").strip().lower() != "disabled"
    }


def _model_selection_request(
    session: dict[str, Any], requested_model: str
) -> tuple[str, dict[str, str]] | None:
    """ACP request selecting ``requested_model`` for ``session``: stable v1
    ``session/set_config_option``, else Copilot's pre-stabilization ``session/set_model``
    when no model config option is advertised. A reported model list is authoritative:
    unknown and policy-disabled ids return None instead of being sent."""
    session_id = str(session.get("sessionId") or "").strip()
    requested_model = str(requested_model or "").strip()
    if not session_id or not requested_model or requested_model == "copilot-acp":
        return None
    options = [
        o
        for o in (session.get("configOptions") or [])
        if isinstance(o, dict) and "model" in (o.get("category"), o.get("id"))
    ]
    if options:
        if requested_model not in _enabled_ids(options[0].get("options"), "value"):
            return None
        return "session/set_config_option", {
            "sessionId": session_id,
            "configId": str(options[0].get("id") or "model"),
            "value": requested_model,
        }
    available = _enabled_ids((session.get("models") or {}).get("availableModels"), "modelId")
    return None if available and requested_model not in available else (
        "session/set_model",
        {"sessionId": session_id, "modelId": requested_model},
    )


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
        return _is_gh_copilot_deprecation_message(stderr_text)

    def _pre_spawn_check(self) -> None:
        """Fast-fail when the CLI doesn't support the ACP args we'd pass."""
        supported = _acp_supported(self._acp_command, self._acp_args)
        if supported is False:
            preview = " ".join(self._acp_args[:3]) if self._acp_args else "(none)"
            raise RuntimeError(
                f"ACP transport not supported by '{self._acp_command}': "
                f"`{preview}` is rejected as an unknown option. "
                f"This usually means the CLI is an older release (e.g. "
                f"Claude Code v2.x) or a different tool than expected. "
                f"Either install a CLI that ships with --acp support "
                f"(e.g. `@github/copilot` late 2025+), or set "
                f"HERMES_COPILOT_ACP_COMMAND / HERMES_COPILOT_ACP_ARGS "
                f"to a working pair."
            )

    def _apply_session_model(
        self,
        session_id: str,
        session: dict[str, Any],
        model: str | None,
        *,
        timeout_seconds: float,
    ) -> None:
        requested_model = str(model or "").strip()
        if not requested_model or requested_model == self._default_model_name:
            return
        try:
            selection = _model_selection_request(session, requested_model)
            if selection is not None:
                self._rpc(*selection, timeout_seconds=timeout_seconds)
            else:
                logger.warning(
                    "Copilot ACP does not offer model %r; using the session default.",
                    requested_model,
                )
        except Exception as exc:
            logger.warning(
                "Copilot ACP model selection for %r failed; continuing with the session default: %s",
                requested_model,
                exc,
            )

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        timeout_seconds: float,
        model: str | None = None,
    ) -> tuple[str, str]:
        return self._run_conversation_prompt(
            [{"role": "user", "content": prompt_text}],
            model=model,
            tools=None,
            tool_choice=None,
            timeout_seconds=timeout_seconds,
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
        if stream:
            return super()._create_chat_completion(
                model=model,
                messages=messages,
                timeout=timeout,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                **kwargs,
            )
        # If _run_prompt was intercepted (e.g. test mocking _run_prompt), route to it.
        if (
            CopilotACPClient._run_prompt is not _DEFAULT_COPILOT_RUN_PROMPT
            or getattr(self._run_prompt, "_mock_self", None) is not None
        ):
            prompt_text = _format_messages_as_prompt(messages)
            timeout_seconds = self._normalize_timeout(timeout)
            model_name = model or self._default_model_name
            response_text, reasoning_text = self._run_prompt(
                prompt_text,
                timeout_seconds=timeout_seconds,
                model=model_name,
            )
            return self._build_completion(
                response_text,
                reasoning_text,
                model_name,
                tools=tools,
            )
        return super()._create_chat_completion(
            model=model,
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )


_DEFAULT_COPILOT_RUN_PROMPT = CopilotACPClient._run_prompt
