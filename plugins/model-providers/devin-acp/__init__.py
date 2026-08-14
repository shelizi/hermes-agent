"""Devin CLI ACP provider profile.

devin-acp uses an external ACP subprocess (``devin acp``) — NOT a REST
chat-completions endpoint. Routing is handled by DevinACPClient, same
pattern as copilot-acp.
"""

from providers import register_provider
from providers.base import ProviderProfile


class DevinACPProfile(ProviderProfile):
    """Devin CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
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
)

register_provider(devin_acp)
