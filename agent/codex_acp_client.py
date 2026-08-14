"""OpenAI-compatible shim that forwards Hermes requests to ``codex-acp``.

Mirrors :class:`agent.acp_client_base.BaseACPClient` for the OpenAI
Codex CLI ACP adapter (``@agentclientprotocol/codex-acp``). The adapter
exposes Codex over stdio JSON-RPC and follows the shared ACP client lifecycle
(see parent module).

Runtime environment:
- ``CODEX_PATH``: path to the native ``codex`` executable. The adapter tries
  to find ``codex`` on PATH, but on Windows the npm wrapper does not reliably
  expose the native binary, so this client resolves and sets it explicitly.
- ``CODEX_API_KEY`` / ``OPENAI_API_KEY``: used by the adapter for the
  ``api-key`` auth method.
- ``NO_BROWSER=1``: hide the ChatGPT browser auth flow in headless mode.

Duck-typed env overrides:
  HERMES_CODEX_ACP_COMMAND / CODEX_ACP_CLI_PATH  -> binary path
  HERMES_CODEX_ACP_ARGS                          -> argv override
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from agent.acp_client_base import BaseACPClient

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://codex"


def _resolve_command() -> str:
    env = (
        os.getenv("HERMES_CODEX_ACP_COMMAND", "").strip()
        or os.getenv("CODEX_ACP_CLI_PATH", "").strip()
    )
    if env:
        return env

    resolved = shutil.which("codex-acp")
    if resolved:
        return resolved

    # Local Hermes-managed install fallback.
    local_cmd = Path.home() / ".hermes" / "codex-acp" / "node_modules" / ".bin" / "codex-acp"
    if local_cmd.is_file():
        return str(local_cmd)

    return "codex-acp"


def _resolve_codex_path(command: str) -> str | None:
    """Return the native ``codex`` executable to advertise to the adapter.

    The adapter needs a native ``codex`` binary, not the npm wrapper script.
    Try the sibling of the command first, then the local Hermes install, then
    PATH.
    """
    # If the command is a local .bin shim, its real binary is usually in the
    # platform package next to the adapter install.
    command_path = Path(command)
    for platform_dir in command_path.parents:
        candidates = list(platform_dir.glob("@openai/codex-*/vendor/*/bin/codex*"))
        if candidates:
            for c in candidates:
                if c.is_file() and c.stem == "codex":
                    return str(c)

    # Local Hermes-managed install.
    local_bin = (
        Path.home()
        / ".hermes"
        / "codex-acp"
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    if local_bin.is_file():
        return str(local_bin)

    # macOS / Linux local install.
    for pattern in (
        "@openai/codex-darwin-x64/vendor/x86_64-apple-darwin/bin/codex",
        "@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex",
        "@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex",
        "@openai/codex-linux-arm64/vendor/aarch64-unknown-linux-musl/bin/codex",
    ):
        local = Path.home() / ".hermes" / "codex-acp" / "node_modules" / ".bin" / ".." / pattern
        local = local.resolve()
        if local.is_file():
            return str(local)

    # Finally, rely on the adapter to resolve via PATH.
    return shutil.which("codex") or None


def _resolve_args(command: str | None = None) -> list[str]:
    raw = os.getenv("HERMES_CODEX_ACP_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    # ``codex-acp`` is already an ACP server; it needs no launch args.
    return []


class CodexACPClient(BaseACPClient):
    """Minimal OpenAI-client-compatible facade for OpenAI Codex CLI ACP."""

    _acp_display_name = "Codex ACP"
    _default_model_name = "codex-acp"
    _install_hint = (
        "Install the Codex ACP adapter (npm install -g @agentclientprotocol/codex-acp "
        "and @openai/codex) or set HERMES_CODEX_ACP_COMMAND/CODEX_ACP_CLI_PATH."
    )
    _acp_marker_base_url = "acp://codex"

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
        self._codex_path: str | None = _resolve_codex_path(self._acp_command)

    def _subprocess_env(self) -> dict[str, str]:
        """Pass the native Codex binary and headless auth hints to the adapter."""
        env = dict(super()._subprocess_env())
        if self._codex_path:
            env["CODEX_PATH"] = self._codex_path
        # Hide browser-based ChatGPT login in headless / automation contexts.
        env["NO_BROWSER"] = "1"
        # Prefer explicit Codex key, fall back to OpenAI key if already set.
        if os.environ.get("CODEX_API_KEY"):
            env["CODEX_API_KEY"] = os.environ["CODEX_API_KEY"]
        elif os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        # Allow JSON config injection for sandbox / model provider defaults.
        if os.environ.get("CODEX_CONFIG"):
            env["CODEX_CONFIG"] = os.environ["CODEX_CONFIG"]
        if os.environ.get("MODEL_PROVIDER"):
            env["MODEL_PROVIDER"] = os.environ["MODEL_PROVIDER"]
        return env

    def _ensure_initialized(self, *, timeout_seconds: float) -> None:
        """Spawn (if needed), run ACP ``initialize`` and ``authenticate``."""
        if self._process_alive() and self._initialized:
            return
        if not self._process_alive():
            self._reset_transport(mark_closed=False)
            self._spawn_process()

        init = self._rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {
                        "readTextFile": True,
                        "writeTextFile": True,
                    }
                },
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            },
            timeout_seconds=timeout_seconds,
        ) or {}
        self._record_initialize_result(init)
        self._initialized = True
        self._authenticate(init, timeout_seconds=timeout_seconds)

    def _authenticate(
        self,
        init: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> None:
        """Call the advertised ``api-key`` auth method if a key is available."""
        auth_methods = init.get("authMethods") or []
        if not isinstance(auth_methods, list) or not auth_methods:
            return

        available_ids = {
            str(m.get("id") or "") for m in auth_methods if isinstance(m, dict)
        }
        if "api-key" not in available_ids:
            return

        api_key = os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return

        # The adapter reads the key from the environment, but the ACP
        # authenticate request still selects the method and may carry metadata.
        auth_meta: dict[str, Any] = {"api-key": {"provider": "openai"}}
        if api_key:
            auth_meta["api-key"]["key"] = api_key

        try:
            self._rpc(
                "authenticate",
                {
                    "methodId": "api-key",
                    "_meta": auth_meta,
                },
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Codex ACP: authenticate(api-key) failed: %s", exc, exc_info=True)
            # Surface auth failure early — session/new will also fail without it.
            raise

    def _prepare_for_model(self, model: str | None) -> None:
        """Codex model selection is currently passed through the adapter env."""
        del model

    def _apply_session_model(
        self,
        session_id: str,
        session: dict[str, Any],
        model: str | None,
        *,
        timeout_seconds: float,
    ) -> None:
        """Hook after ``session/new`` to bind the Hermes-selected model.

        Default no-op. The codex-acp adapter currently picks the model from
        ``MODEL_PROVIDER`` / ``CODEX_CONFIG`` env or the Codex default; a future
        ACP ``session/set_config_option`` or ``session/set_model`` can be added
        here once the adapter documents it.
        """
        del session_id, session, model, timeout_seconds

    def _spawn_argv(self) -> list[str]:
        """Wrap npm ``.cmd`` shims so they can be spawned on Windows.

        ``subprocess.Popen`` on Windows requires a PE executable when
        ``shell=False``. The ``codex-acp`` npm package ships a ``.cmd`` shim
        that delegates to ``node``, so we spawn it through ``cmd /c``.
        """
        argv = [self._acp_command] + list(self._acp_args)
        if os.name != "nt":
            return argv

        ext = Path(self._acp_command).suffix.lower()
        is_batch = self._acp_command.lower().endswith(".cmd") or ext in {
            ".cmd",
            ".bat",
            ".ps1",
        }
        # Some package-manager shims have no extension on Windows (e.g. scoop
        # ``codex-acp``); the shebang is ignored by CreateProcess, so treat
        # those as batch as well.
        if not ext and "." not in Path(self._acp_command).name:
            is_batch = True

        if is_batch:
            return ["cmd", "/c", self._acp_command] + list(self._acp_args)
        return argv

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
            model=model or "codex-acp",
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )
