"""Tests for Google Antigravity CLI ACP provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.antigravity_acp_client import (
    ACP_MARKER_BASE_URL,
    AntigravityACPClient,
    _resolve_args,
)
from agent.acp_client_factory import is_acp_provider, create_acp_client
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    resolve_external_process_provider_credentials,
)


class TestAntigravityAcpProviderRegistry(unittest.TestCase):
    def test_registry_entry(self):
        p = PROVIDER_REGISTRY["antigravity-acp"]
        assert p.auth_type == "external_process"
        assert p.inference_base_url == "acp://antigravity"

    def test_aliases(self):
        assert resolve_provider("antigravity") == "antigravity-acp"
        assert resolve_provider("antigravity-cli") == "antigravity-acp"
        assert resolve_provider("google-antigravity") == "antigravity-acp"
        assert resolve_provider("google-antigravity-cli") == "antigravity-acp"

    def test_is_acp_provider(self):
        assert is_acp_provider("antigravity-acp")
        assert is_acp_provider(base_url="acp://antigravity")


class TestAntigravityAcpClientDefaults(unittest.TestCase):
    def test_marker_and_defaults(self):
        assert ACP_MARKER_BASE_URL == "acp://antigravity"
        assert _resolve_args() == ["--acp"]
        client = AntigravityACPClient(
            acp_cwd="/tmp",
            command="agy",
            args=["--acp"],
        )
        assert client.api_key == "antigravity-acp"
        assert client.base_url == "acp://antigravity"
        assert client._acp_command == "agy"
        assert client._acp_args == ["--acp"]

    def test_create_acp_client_factory_returns_antigravity_client(self):
        client = create_acp_client(provider="antigravity-acp")
        assert isinstance(client, AntigravityACPClient)

    def test_resolve_args_env_override(self):
        with patch.dict(
            "os.environ",
            {"HERMES_ANTIGRAVITY_ACP_ARGS": "agent stdio --debug"},
            clear=False,
        ):
            assert _resolve_args() == ["agent", "stdio", "--debug"]

    def test_resolve_command_prefers_env(self):
        with patch.dict(
            "os.environ",
            {
                "HERMES_ANTIGRAVITY_ACP_COMMAND": "/opt/antigravity/bin/antigravity",
            },
            clear=False,
        ):
            client = AntigravityACPClient()
            assert client._acp_command == "/opt/antigravity/bin/antigravity"

    def test_resolve_args_native_agy_gets_acp_flag(self):
        assert _resolve_args("agy") == ["--acp"]
        assert _resolve_args("C:\\Program Files\\antigravity\\bin\\agy.exe") == ["--acp"]

    def test_resolve_args_adapter_gets_no_default_flags(self):
        assert _resolve_args("agy-acp") == []
        assert _resolve_args("/usr/local/bin/antigravity-acp") == []

    def test_resolve_args_from_credentials(self):
        with patch(
            "hermes_cli.auth.shutil.which",
            return_value="/usr/local/bin/agy",
        ):
            with patch(
                "hermes_cli.auth._external_process_auth_present",
                return_value=None,
            ):
                creds = resolve_external_process_provider_credentials(
                    "antigravity-acp"
                )
                assert creds["provider"] == "antigravity-acp"
                assert creds["api_key"] == "antigravity-acp"
                assert creds["base_url"] == "acp://antigravity"
                assert creds["command"] == "/usr/local/bin/agy"
                assert creds["args"] == ["--acp"]
