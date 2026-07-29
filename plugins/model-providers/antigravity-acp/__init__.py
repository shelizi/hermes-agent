"""Google Antigravity CLI ACP provider profile.

antigravity-acp uses an external ACP subprocess
(``antigravity acp``) — NOT a REST chat-completions endpoint. Routing is
handled by :class:`agent.antigravity_acp_client.AntigravityACPClient`, same
pattern as copilot-acp, devin-acp and grok-acp.
"""

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


antigravity_acp = AntigravityACPProfile(
    name="antigravity-acp",
    display_name="Google Antigravity CLI ACP",
    description="Google Antigravity CLI via ACP (antigravity acp).",
    aliases=("antigravity-cli", "google-antigravity", "google-antigravity-cli"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://antigravity",
    auth_type="external_process",
    # Keep OpenAI-style image_url parts on the user turn so AntigravityACPClient
    # can re-encode them as ACP content blocks when the CLI advertises
    # promptCapabilities.image (see agent/copilot_acp_client.py).
    supports_vision=True,
)

register_provider(antigravity_acp)
