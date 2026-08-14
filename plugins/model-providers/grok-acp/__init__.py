"""Grok Build CLI ACP provider profile.

grok-acp uses an external ACP subprocess
(``grok --no-auto-update agent stdio``) — NOT a REST chat-completions
endpoint. Routing is handled by GrokACPClient, same pattern as
copilot-acp and devin-acp.

Official ACP entrypoint (xAI docs / Zed ACP registry):
  https://docs.x.ai/build/cli/headless-scripting
  ``grok agent stdio``
"""

import subprocess
from pathlib import Path

from providers import register_provider
from providers.base import ProviderProfile


class GrokACPProfile(ProviderProfile):
    """Grok Build CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 12.0,
    ) -> list[str] | None:
        """Discover Grok CLI models by running ``grok models``."""
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
                [resolved, "models"],
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
            from hermes_cli.models import parse_grok_cli_available_models

            return parse_grok_cli_available_models(blob)
        except Exception:
            return []

    def auth_present(self) -> bool | None:
        """Grok CLI uses XAI_API_KEY or ~/.grok/auth.json."""
        try:
            from hermes_cli.auth import _grok_local_credentials_present

            return _grok_local_credentials_present()
        except Exception:
            return None

    def search_command_path(self, command: str) -> str | None:
        """Grok Build CLI installs under ~/.grok/bin."""
        command_name = Path(command).name.casefold()
        if command_name not in {"grok", "grok.exe"}:
            return None

        home = Path.home()
        for candidate in (
            home / ".grok" / "bin" / "grok.exe",
            home / ".grok" / "bin" / "grok",
        ):
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:
                continue
        return None


grok_acp = GrokACPProfile(
    name="grok-acp",
    display_name="Grok CLI ACP",
    description="Grok Build CLI via ACP (grok --no-auto-update agent stdio).",
    aliases=("grok-cli", "grok-build", "xai-grok-cli"),
    api_mode="chat_completions",
    env_vars=("GROK_ACP_BASE_URL",),
    base_url="acp://grok",
    auth_type="external_process",
    # Keep OpenAI-style image_url parts on the user turn so GrokACPClient can
    # re-encode them as ACP ``image`` content blocks when the CLI advertises
    # promptCapabilities.image (see agent/copilot_acp_client.py).
    supports_vision=True,
    process_spec={
        "command_env": ("HERMES_GROK_ACP_COMMAND", "GROK_CLI_PATH"),
        "default_command": "grok",
        "args_env": "HERMES_GROK_ACP_ARGS",
        # Official ACP: `grok agent stdio`; --no-auto-update for automation.
        "default_args": ["--no-auto-update", "agent", "stdio"],
        "api_key": "grok-acp",
        "missing_code": "missing_grok_cli",
        "missing_msg": (
            "Could not find the Grok CLI command '{command}'. "
            "Install Grok Build CLI (https://docs.x.ai/build/cli), run `grok login`, "
            "or set HERMES_GROK_ACP_COMMAND/GROK_CLI_PATH."
        ),
        "login_hint": "Grok CLI found but no local credentials — run: grok login or set XAI_API_KEY",
    },
    fallback_models=(
        "grok-4.5",
        "grok-composer-2.5-fast",
        "grok-build-0.1",
        "grok-4.3",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-multi-agent-0309",
        "grok-3-mini",
        "grok-3-mini-fast",
        # Hermes placeholder when no specific model is chosen
        "grok-acp",
    ),
)

register_provider(grok_acp)
