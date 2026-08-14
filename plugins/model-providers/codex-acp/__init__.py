"""OpenAI Codex CLI ACP provider profile.

codex-acp uses an external ACP subprocess (``codex-acp``) — NOT a REST
chat-completions endpoint. Routing is handled by CodexACPClient, same pattern
as copilot-acp, devin-acp, grok-acp and antigravity-acp.
"""

from providers import register_provider
from providers.base import ProviderProfile


class CodexACPProfile(ProviderProfile):
    """OpenAI Codex CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess or hermes_cli.models."""
        return None


codex_acp = CodexACPProfile(
    name="codex-acp",
    display_name="OpenAI Codex CLI ACP",
    description="OpenAI Codex CLI via ACP (codex-acp).",
    aliases=("codex-cli", "openai-codex-acp"),
    api_mode="chat_completions",
    env_vars=("CODEX_ACP_BASE_URL",),
    base_url="acp://codex",
    auth_type="external_process",
    # Keep OpenAI-style image_url parts on the user turn so CodexACPClient can
    # re-encode them as ACP content blocks when the CLI advertises
    # promptCapabilities.image.
    supports_vision=True,
    process_spec={
        "command_env": ("HERMES_CODEX_ACP_COMMAND", "CODEX_ACP_CLI_PATH"),
        "default_command": "codex-acp",
        "args_env": "HERMES_CODEX_ACP_ARGS",
        # ``codex-acp`` is an ACP server; it needs no launch args.
        "default_args": [],
        "api_key": "codex-acp",
        "missing_code": "missing_codex_acp_cli",
        "missing_msg": (
            "Could not find the Codex ACP command '{command}'. "
            "Install @agentclientprotocol/codex-acp and @openai/codex "
            "or set HERMES_CODEX_ACP_COMMAND/CODEX_ACP_CLI_PATH."
        ),
    },
)

register_provider(codex_acp)
