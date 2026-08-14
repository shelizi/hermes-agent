"""GitHub Copilot ACP provider profile.

copilot-acp uses an external ACP subprocess — NOT the standard
transport. api_mode="copilot_acp" is handled separately in run_agent.py.
The profile captures auth + endpoint metadata for registry migration.
"""

from providers import register_provider
from providers.base import ProviderProfile


class CopilotACPProfile(ProviderProfile):
    """GitHub Copilot ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


copilot_acp = CopilotACPProfile(
    name="copilot-acp",
    display_name="GitHub Copilot ACP",
    description="GitHub Copilot CLI via ACP (copilot --acp --stdio).",
    aliases=("github-copilot-acp", "copilot-acp-agent"),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=("COPILOT_ACP_BASE_URL",),
    base_url="acp://copilot",  # ACP internal scheme
    auth_type="external_process",
    # Native image_url parts are re-encoded as ACP content blocks when the
    # Copilot agent advertises promptCapabilities.image.
    supports_vision=True,
    process_spec={
        "command_env": ("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
        "default_command": "copilot",
        "args_env": "HERMES_COPILOT_ACP_ARGS",
        "default_args": ["--acp", "--stdio"],
        "api_key": "copilot-acp",
        "missing_code": "missing_copilot_cli",
        "missing_msg": (
            "Could not find the Copilot CLI command '{command}'. "
            "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
        ),
    },
)

register_provider(copilot_acp)
