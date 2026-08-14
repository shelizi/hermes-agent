"""Factory helpers for ACP subprocess-backed chat clients."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

# Lazy dispatch table: provider id -> (module_name, class_name).
# We keep strings rather than class objects so that runtime patches
# (e.g. tests that mock CopilotACPClient) are respected on every call.
_ACP_CLIENT_TABLE: dict[str, tuple[str, str]] | None = None


def _provider_id_from_inputs(
    provider: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """Return a canonical-ish provider id from the provider name or base_url."""
    p = (provider or "").strip().lower()
    if p:
        return p

    url = (base_url or "").strip().lower()
    if not url:
        return None

    if url.startswith("acp://") or url.startswith("acp+tcp://"):
        return (urlparse(url).hostname or "").strip() or None

    return None


def _acp_client_table() -> dict[str, tuple[str, str]]:
    """Return the provider-id -> (module, class) dispatch table."""
    global _ACP_CLIENT_TABLE
    if _ACP_CLIENT_TABLE is None:
        _ACP_CLIENT_TABLE = {
            "antigravity-acp": ("agent.antigravity_acp_client", "AntigravityACPClient"),
            "codex-acp": ("agent.codex_acp_client", "CodexACPClient"),
            "copilot-acp": ("agent.copilot_acp_client", "CopilotACPClient"),
            "devin-acp": ("agent.devin_acp_client", "DevinACPClient"),
            "grok-acp": ("agent.grok_acp_client", "GrokACPClient"),
        }
    return _ACP_CLIENT_TABLE


def _resolve_acp_client_module(provider_id: str) -> tuple[str, str] | None:
    """Return the (module, class) spec for *provider_id*, or None."""
    table = _acp_client_table()
    if provider_id in table:
        return table[provider_id]

    # Allow bare host names from acp://<host> URLs to map onto <host>-acp.
    if not provider_id.endswith("-acp"):
        return table.get(provider_id + "-acp")

    return None


def _acp_client_class(provider_id: str) -> type | None:
    """Return the ACP client class for *provider_id*, or None if unknown.

    Import is done on every call so that runtime monkey-patches (tests,
    emergency overrides) are always respected and we don't cache a concrete
    class object across calls.
    """
    spec = _resolve_acp_client_module(provider_id)
    if spec is None:
        return None

    module_name, class_name = spec
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None


def is_acp_provider(
    provider: str | None = None,
    base_url: str | None = None,
) -> bool:
    """Return True when *provider* or *base_url* points at an ACP subprocess backend."""
    p = _provider_id_from_inputs(provider, base_url)
    if p and _resolve_acp_client_module(p):
        return True

    # Any base_url using the ACP scheme is an ACP backend, even if we don't
    # have a dedicated client class for it yet.
    url = (base_url or "").strip().lower()
    if url.startswith("acp://") or url.startswith("acp+tcp://"):
        return True

    # Fallback: the Hermes provider registry or plugin profile marks it as an
    # external-process provider and its base_url is an ACP scheme.
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get((provider or "").strip().lower())
        if pconfig and getattr(pconfig, "auth_type", None) == "external_process":
            return True
    except Exception:
        pass

    try:
        from providers import get_provider_profile

        profile = get_provider_profile((provider or "").strip().lower())
        if getattr(profile, "auth_type", None) == "external_process":
            return True
        if str(getattr(profile, "base_url", "")).lower().startswith("acp://"):
            return True
    except Exception:
        pass

    return False


def _fill_missing_acp_invocation(
    provider: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Fill empty command/args from the shared auth resolver.

    Call sites sometimes construct an ACP client with only provider/base_url.
    When that happens, prefer the shared auth resolver (env + PATH) over each
    client's private defaults -- keeps command/args consistent with
    ``resolve_runtime_provider`` / ``hermes status``.
    """
    command = kwargs.get("command") or kwargs.get("acp_command")
    raw_args = kwargs.get("args")
    if raw_args is None:
        raw_args = kwargs.get("acp_args")
    has_args = isinstance(raw_args, (list, tuple)) and len(raw_args) > 0

    if command and has_args:
        return kwargs

    try:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        creds = resolve_external_process_provider_credentials(provider)
    except Exception:
        return kwargs

    filled = dict(kwargs)
    if not command:
        resolved = str(creds.get("command") or "").strip()
        if resolved:
            filled["command"] = resolved
    if not has_args:
        cred_args = list(creds.get("args") or [])
        if cred_args or "args" not in filled:
            filled["args"] = cred_args
    return filled


def create_acp_client(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate the correct ACP client for *provider* / *base_url*."""
    p = _provider_id_from_inputs(provider, base_url)
    if not p:
        p = "copilot-acp"

    client_class = _acp_client_class(p)
    if client_class is None:
        from agent.copilot_acp_client import CopilotACPClient

        client_class = CopilotACPClient

    client_kwargs = _fill_missing_acp_invocation(p, kwargs)
    return client_class(provider=p, base_url=base_url, **client_kwargs)
