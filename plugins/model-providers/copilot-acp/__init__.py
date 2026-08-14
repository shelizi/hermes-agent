"""GitHub Copilot ACP provider profile.

copilot-acp uses an external ACP subprocess — NOT the standard
transport. api_mode="copilot_acp" is handled separately in run_agent.py.
The profile captures auth + endpoint metadata for registry migration.
"""

from typing import Any

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
)

register_provider(copilot_acp)
