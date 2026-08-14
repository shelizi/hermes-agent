"""Tests for Grok Build CLI ACP provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import providers
from agent.grok_acp_client import (
    ACP_MARKER_BASE_URL,
    GrokACPClient,
    _resolve_args,
    resolve_grok_acp_model_value,
)
from agent.acp_client_factory import is_acp_provider, create_acp_client
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
    resolve_provider,
)


class TestGrokAcpProviderRegistry(unittest.TestCase):
    def test_registry_entry(self):
        p = PROVIDER_REGISTRY["grok-acp"]
        assert p.auth_type == "external_process"
        assert p.inference_base_url == "acp://grok"

    def test_aliases(self):
        # Direct xAI API keeps the short "grok" alias; CLI ACP uses explicit names.
        assert resolve_provider("grok") == "xai"
        assert resolve_provider("grok-cli") == "grok-acp"
        assert resolve_provider("grok-build") == "grok-acp"
        assert resolve_provider("xai-grok-cli") == "grok-acp"

    def test_is_acp_provider(self):
        assert is_acp_provider("grok-acp")
        assert is_acp_provider(base_url="acp://grok")


class TestGrokAcpResolve(unittest.TestCase):
    def test_status_and_creds(self):
        grok = providers.get_provider_profile("grok-acp")
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/grok"):
            with patch.object(grok, "auth_present", return_value=True):
                with patch.dict(
                    "os.environ",
                    {"HERMES_GROK_ACP_ARGS": "--no-auto-update agent stdio"},
                    clear=False,
                ):
                    status = get_external_process_provider_status("grok-acp")
                    assert status["configured"] is True
                    assert status["cli_installed"] is True
                    assert status["auth_present"] is True
                    assert status["logged_in"] is True
                    assert status["command"] == "grok"
                    assert status["resolved_command"] == "/usr/local/bin/grok"
                    assert status["args"] == ["--no-auto-update", "agent", "stdio"]
                    assert status["base_url"] == "acp://grok"

                    creds = resolve_external_process_provider_credentials("grok-acp")
                    assert creds["provider"] == "grok-acp"
                    assert creds["api_key"] == "grok-acp"
                    assert creds["base_url"] == "acp://grok"
                    assert creds["command"] == "/usr/local/bin/grok"
                    assert creds["args"] == ["--no-auto-update", "agent", "stdio"]

    def test_status_cli_without_credentials_is_not_logged_in(self):
        grok = providers.get_provider_profile("grok-acp")
        with patch("hermes_cli.auth.shutil.which", return_value="/usr/local/bin/grok"):
            with patch.object(grok, "auth_present", return_value=False):
                status = get_external_process_provider_status("grok-acp")
        assert status["configured"] is True
        assert status["cli_installed"] is True
        assert status["auth_present"] is False
        assert status["logged_in"] is False
        assert status.get("hint")
        assert "grok login" in status["hint"]


class TestGrokAcpClientDefaults(unittest.TestCase):
    def test_marker_and_defaults(self):
        assert ACP_MARKER_BASE_URL == "acp://grok"
        with patch.dict("os.environ", {"HERMES_GROK_ACP_ARGS": ""}, clear=False):
            assert _resolve_args() == ["--no-auto-update", "agent", "stdio"]
        client = GrokACPClient(
            acp_cwd="/tmp",
            command="grok",
            args=["--no-auto-update", "agent", "stdio"],
        )
        assert client.api_key == "grok-acp"
        assert client.base_url == "acp://grok"
        assert client._acp_command == "grok"
        assert client._acp_args == ["--no-auto-update", "agent", "stdio"]

    def test_backend_model_id_maps_placeholders_to_none(self):
        from agent.grok_acp_client import _backend_model_id

        assert _backend_model_id(None) is None
        assert _backend_model_id("") is None
        assert _backend_model_id("grok-acp") is None
        assert _backend_model_id("grok-cli") is None
        assert _backend_model_id("grok-build") is None
        assert _backend_model_id("grok-4.5") == "grok-4.5"

    def test_resolve_grok_acp_model_value_maps_cli_to_acp_ids(self):
        available = [
            {"modelId": "grok-4.5", "name": "Grok 4.5"},
            {"modelId": "grok-composer-2.5-fast", "name": "Grok Composer 2.5 Fast"},
            {"modelId": "grok-build-0.1", "name": "Grok Build 0.1"},
        ]
        assert resolve_grok_acp_model_value("grok-4.5", available) == "grok-4.5"
        assert (
            resolve_grok_acp_model_value("grok-composer-2.5-fast", available)
            == "grok-composer-2.5-fast"
        )
        assert resolve_grok_acp_model_value("grok-4-5", available) == "grok-4.5"
        assert resolve_grok_acp_model_value("grok-build-0.1", available) == "grok-build-0.1"
        assert resolve_grok_acp_model_value("grok-acp", available) is None

    def test_create_acp_client_factory_returns_grok_client(self):
        client = create_acp_client(provider="grok-acp")
        assert isinstance(client, GrokACPClient)


class TestGrokAcpHermesMcpBridge(unittest.TestCase):
    """Hermes memory + tools MCP servers attached like Devin ACP."""

    def test_memory_mcp_bridge_is_attached_only_when_memory_is_granted(self):
        client = GrokACPClient(
            acp_cwd="/tmp",
            command="grok",
            args=["--no-auto-update", "agent", "stdio"],
        )
        memory_tools = [
            {
                "type": "function",
                "function": {"name": "memory"},
            }
        ]
        with patch(
            "agent.transports.hermes_memory_mcp_server.build_acp_server_config",
            return_value=[{"name": "hermes-memory"}],
        ) as build:
            assert client._session_mcp_servers([]) == []
            assert client._session_mcp_servers(memory_tools) == [
                {"name": "hermes-memory"}
            ]
        build.assert_called_once_with()

    def test_hermes_tools_mcp_bridge_uses_granted_tool_names(self):
        client = GrokACPClient(
            acp_cwd="/tmp",
            command="grok",
            args=["--no-auto-update", "agent", "stdio"],
        )
        tools = [
            {"type": "function", "function": {"name": "skills_list"}},
            {"type": "function", "function": {"name": "skill_view"}},
            {"type": "function", "function": {"name": "skill_manage"}},
            {"type": "function", "function": {"name": "todo"}},
            {"type": "function", "function": {"name": "session_search"}},
        ]
        with patch(
            "agent.transports.hermes_tools_mcp_server.build_acp_server_config",
            return_value=[{"name": "hermes-tools"}],
        ) as build:
            assert client._session_mcp_servers(tools) == [{"name": "hermes-tools"}]

        build.assert_called_once()
        assert set(build.call_args.kwargs["allowed_tools"]) == {
            "skills_list",
            "skill_view",
            "skill_manage",
            "todo",
            "session_search",
        }

    def test_native_mcp_tools_are_not_duplicated_in_text_prompt(self):
        client = GrokACPClient(
            acp_cwd="/tmp",
            command="grok",
            args=["--no-auto-update", "agent", "stdio"],
        )
        tools = [{"type": "function", "function": {"name": "skill_view"}}]
        assert client._prompt_tools(tools) is None

    def test_native_mcp_bridge_exposes_all_tools_without_prompt_schema(self):
        client = GrokACPClient(
            acp_cwd="/tmp",
            command="grok",
            args=["--no-auto-update", "agent", "stdio"],
        )
        with patch(
            "agent.transports.hermes_memory_mcp_server.build_acp_server_config",
            return_value=[{"name": "hermes-memory"}],
        ) as memory_build, patch(
            "agent.transports.hermes_tools_mcp_server.build_acp_server_config",
            return_value=[{"name": "hermes-tools"}],
        ) as tools_build:
            assert client._session_mcp_servers(None) == [
                {"name": "hermes-memory"},
                {"name": "hermes-tools"},
            ]

        memory_build.assert_called_once_with()
        tools_build.assert_called_once_with()


class TestGrokAcpImageContentParts(unittest.TestCase):
    """Grok ACP records promptCapabilities and forwards image content parts."""

    def test_initialize_records_image_prompt_capability(self):
        client = GrokACPClient(
            acp_cwd="/tmp",
            command="grok",
            args=["--no-auto-update", "agent", "stdio"],
        )
        # Pretend the process is already warm so _ensure_initialized only RPCs.
        client._initialized = False

        def fake_alive():
            return True

        def fake_rpc(method, params, **kwargs):
            assert method == "initialize"
            return {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "promptCapabilities": {"image": True, "audio": False},
                },
                "authMethods": [],
            }

        with patch.object(client, "_process_alive", side_effect=fake_alive), patch.object(
            client, "_rpc", side_effect=fake_rpc
        ):
            client._ensure_initialized(timeout_seconds=5)

        assert client._initialized is True
        assert client._prompt_capabilities["image"] is True
        assert client._prompt_capabilities["audio"] is False

    def test_image_routing_treats_grok_acp_as_vision_capable(self):
        from agent.image_routing import _lookup_supports_vision

        assert _lookup_supports_vision("grok-acp", "grok-4.5", {}) is True
        assert _lookup_supports_vision("devin-acp", "devin", {}) is True
        assert _lookup_supports_vision("copilot-acp", "copilot-acp", {}) is True


if __name__ == "__main__":
    unittest.main()
