"""Google Antigravity CLI ACP provider profile.

antigravity-acp uses an external ACP subprocess
(``antigravity acp``) — NOT a REST chat-completions endpoint. Routing is
handled by :class:`agent.antigravity_acp_client.AntigravityACPClient`, same
pattern as copilot-acp, devin-acp and grok-acp.
"""

from pathlib import Path

from providers import register_provider
from providers.base import ProviderProfile


class AntigravityACPProfile(ProviderProfile):
    """Google Antigravity CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess or hermes_cli.models."""
        return None

    def search_command_path(self, command: str) -> str | None:
        """Antigravity CLI installs under ~/.antigravity/bin."""
        command_name = Path(command).name.casefold()
        if command_name not in {"agy", "agy.exe"}:
            return None

        home = Path.home()
        for candidate in (
            home / ".antigravity" / "bin" / "agy.exe",
            home / ".antigravity" / "bin" / "agy",
        ):
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:
                continue
        return None

    def resolve_command_args(self, command: str, default_args: list[str]) -> list[str]:
        """Adapter binaries (agy-acp, antigravity-acp, etc.) are already ACP servers.

        Only the native ``agy`` binary needs the ``--acp`` launch flag.
        """
        command_name = Path(command).name.casefold()
        if command_name not in {"agy", "agy.exe", "antigravity", "antigravity.exe"}:
            return []
        return list(default_args)


antigravity_acp = AntigravityACPProfile(
    name="antigravity-acp",
    display_name="Google Antigravity CLI ACP",
    description="Google Antigravity CLI via ACP (antigravity --acp).",
    aliases=("antigravity", "antigravity-cli", "google-antigravity", "google-antigravity-cli"),
    api_mode="chat_completions",
    env_vars=("ANTIGRAVITY_ACP_BASE_URL",),
    base_url="acp://antigravity",
    auth_type="external_process",
    # Keep OpenAI-style image_url parts on the user turn so AntigravityACPClient
    # can re-encode them as ACP content blocks when the CLI advertises
    # promptCapabilities.image (see agent/copilot_acp_client.py).
    supports_vision=True,
    process_spec={
        "command_env": ("HERMES_ANTIGRAVITY_ACP_COMMAND", "ANTIGRAVITY_CLI_PATH"),
        "default_command": "agy",
        "args_env": "HERMES_ANTIGRAVITY_ACP_ARGS",
        # Native ACP entrypoint is ``agy --acp`` (not yet shipped as of 1.0.16).
        # Until then, override this to an adapter binary such as ``agy-acp``.
        "default_args": ["--acp"],
        "api_key": "antigravity-acp",
        "missing_code": "missing_antigravity_cli",
        "missing_msg": (
            "Could not find the Antigravity CLI command '{command}'. "
            "Install Google Antigravity CLI (agy) or an ACP adapter (agy-acp), "
            "or set HERMES_ANTIGRAVITY_ACP_COMMAND/ANTIGRAVITY_CLI_PATH."
        ),
    },
    fallback_models=("antigravity-acp",),
)

register_provider(antigravity_acp)
