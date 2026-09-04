"""GitHub Copilot ACP provider profile.

copilot-acp does not speak OpenAI-over-HTTP: it drives an external ACP subprocess over
stdio, so the profile supplies its own client via :meth:`ProviderProfile.create_client`.
An out-of-tree ACP provider (``~/.hermes/plugins/model-providers/`` or a pip entry point)
uses the same three lines without touching core.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CopilotACPProfile(ProviderProfile):
    """GitHub Copilot ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the ACP stdio shim rather than an HTTP client."""
        from agent.copilot_acp_client import CopilotACPClient

        return CopilotACPClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Fetch GitHub Copilot model catalog to populate the picker.

        Falls back to the ACP placeholder only when the catalog cannot be
        reached, so the picker is still usable in offline/no-auth scenarios.
        """
        try:
            from hermes_cli.models import _PROVIDER_MODELS, _fetch_github_models
            from hermes_cli.auth import resolve_api_key_provider_credentials

            fallback = list(_PROVIDER_MODELS.get("copilot", []))
            try:
                creds = resolve_api_key_provider_credentials("copilot")
                catalog_key = str(creds.get("api_key") or "").strip()
            except Exception:
                catalog_key = ""
            if not catalog_key:
                return fallback or None
            live = _fetch_github_models(catalog_key, timeout=timeout)
            return live or fallback or None
        except Exception:
            return None

    def normalize_model_id(
        self, model_id: str | None, *, catalog: list[str] | None = None, **context: Any
    ) -> str:
        """Resolve Copilot model aliases against the live GitHub catalog."""
        from hermes_cli.models import normalize_copilot_model_id
        from hermes_cli.auth import resolve_api_key_provider_credentials

        api_key = context.get("api_key")
        if not api_key:
            try:
                creds = resolve_api_key_provider_credentials("copilot")
                api_key = str(creds.get("api_key") or "").strip()
            except Exception:
                api_key = None
        return normalize_copilot_model_id(
            model_id,
            catalog=None,
            api_key=api_key or None,
        )


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
    fallback_models=("copilot-acp",),
    # How to launch the CLI; env var names predate this profile (formerly hardcoded in
    # hermes_cli/auth.py), so existing setups keep working.
    process_command="copilot",
    process_args=("--acp", "--stdio"),
    process_command_env_vars=("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
    process_args_env_var="HERMES_COPILOT_ACP_ARGS",
)

register_provider(copilot_acp)
