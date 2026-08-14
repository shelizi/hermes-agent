"""Devin CLI ACP provider profile.

devin-acp uses an external ACP subprocess (``devin acp``) — NOT a REST
chat-completions endpoint. Routing is handled by DevinACPClient, same
pattern as copilot-acp.
"""

import os
import subprocess
from pathlib import Path

from providers import register_provider
from providers.base import ProviderProfile


# Devin CLI does not expose a REST /models endpoint or `devin models` subcommand.
# Probing with an invalid --model prints: "Available: a, b, c" on stderr.
_DEVIN_PROBE_MODEL = "__hermes_devin_probe__"


class DevinACPProfile(ProviderProfile):
    """Devin CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 12.0,
    ) -> list[str] | None:
        """Discover Devin CLI models by probing ``devin --model <invalid> -p .``."""
        try:
            from hermes_cli.auth import (
                _resolve_external_process_command_args,
                _resolve_external_process_command_path,
            )

            command, _ = _resolve_external_process_command_args(self.name)
            resolved = _resolve_external_process_command_path(self.name, command)
        except Exception:
            return []

        if not resolved:
            return []

        try:
            proc = subprocess.run(
                [resolved, "--model", _DEVIN_PROBE_MODEL, "-p", "."],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []
        except Exception:
            return []

        blob = "\n".join(
            part for part in (proc.stderr or "", proc.stdout or "") if part
        )
        try:
            from hermes_cli.models import parse_devin_cli_available_models

            return parse_devin_cli_available_models(blob)
        except Exception:
            return []

    def auth_present(self) -> bool | None:
        """Devin CLI writes a local credentials.toml; probe it without loading secrets."""
        try:
            from hermes_cli.auth import _devin_local_credentials_present

            return _devin_local_credentials_present()
        except Exception:
            return None

    def search_command_path(self, command: str) -> str | None:
        r"""Devin CLI installer keeps the binary under %LOCALAPPDATA%\devin\cli\bin."""
        command_name = Path(command).name.casefold()
        if command_name not in {"devin", "devin.exe"}:
            return None

        roots: list[Path] = []
        for env_name in ("LOCALAPPDATA", "APPDATA"):
            raw_root = os.getenv(env_name, "").strip()
            if raw_root:
                roots.append(Path(raw_root))
        roots.append(Path.home() / "AppData" / "Local")

        seen: set[str] = set()
        for root in roots:
            candidate = root / "devin" / "cli" / "bin" / "devin.exe"
            key = str(candidate).casefold()
            if key in seen:
                continue
            seen.add(key)
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:
                continue
        return None


devin_acp = DevinACPProfile(
    name="devin-acp",
    display_name="Devin CLI ACP",
    description="Devin CLI via ACP (devin acp).",
    aliases=("devin", "devin-cli", "cognition-devin"),
    api_mode="chat_completions",
    env_vars=("DEVIN_ACP_BASE_URL",),
    base_url="acp://devin",
    auth_type="external_process",
    # Native image_url parts are re-encoded as ACP content blocks when the
    # Devin agent advertises promptCapabilities.image.
    supports_vision=True,
    process_spec={
        "command_env": ("HERMES_DEVIN_ACP_COMMAND", "DEVIN_CLI_PATH"),
        "default_command": "devin",
        "args_env": "HERMES_DEVIN_ACP_ARGS",
        "default_args": ["acp"],
        "api_key": "devin-acp",
        "missing_code": "missing_devin_cli",
        "missing_msg": (
            "Could not find the Devin CLI command '{command}'. "
            "Install Devin CLI (https://docs.devin.ai/cli), run `devin auth login`, "
            "or set HERMES_DEVIN_ACP_COMMAND/DEVIN_CLI_PATH."
        ),
        "login_hint": "Devin CLI found but no local credentials — run: devin auth login",
    },
    fallback_models=(
        "adaptive",
        "swe-1.7",
        "swe-1.7-lightning",
        "swe-1.6",
        "swe-1.6-fast",
        "swe-1.5",
        "claude-opus-4.8",
        "claude-opus-4.7",
        "claude-opus-4.5",
        "claude-sonnet-5",
        "claude-sonnet-4.6",
        "claude-sonnet-4.5",
        "claude-fable-5",
        "claude-haiku-4.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.2",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "gemini-3-flash",
        "grok-4.5",
        "deepseek-v4-pro",
        "kimi-k2.7",
        "kimi-k2.6",
        "glm-5.2",
        "nemotron-3-ultra",
        # Family aliases (docs: always resolve to latest in family)
        "opus",
        "sonnet",
        "swe",
        "codex",
        "gemini",
        # Hermes placeholder when no specific model is chosen
        "devin-acp",
    ),
)

register_provider(devin_acp)
