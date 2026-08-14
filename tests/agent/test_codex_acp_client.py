"""Tests for OpenAI Codex CLI ACP provider wiring."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.codex_acp_client import (
    ACP_MARKER_BASE_URL,
    CodexACPClient,
    _resolve_args,
    _resolve_command,
    _resolve_codex_path,
)
from agent.acp_client_factory import is_acp_provider, create_acp_client
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    resolve_external_process_provider_credentials,
)


class TestCodexAcpProviderRegistry(unittest.TestCase):
    def test_registry_entry(self):
        p = PROVIDER_REGISTRY["codex-acp"]
        assert p.auth_type == "external_process"
        assert p.inference_base_url == "acp://codex"

    def test_aliases(self):
        assert resolve_provider("codex-cli") == "codex-acp"
        assert resolve_provider("openai-codex-acp") == "codex-acp"

    def test_is_acp_provider(self):
        assert is_acp_provider("codex-acp")
        assert is_acp_provider(base_url="acp://codex")


class TestCodexAcpClientDefaults(unittest.TestCase):
    def test_marker_and_defaults(self):
        assert ACP_MARKER_BASE_URL == "acp://codex"
        assert _resolve_args() == []

    def test_create_acp_client_factory_returns_codex_client(self):
        client = create_acp_client(provider="codex-acp")
        assert isinstance(client, CodexACPClient)

    def test_resolve_args_env_override(self):
        with patch.dict(
            "os.environ",
            {"HERMES_CODEX_ACP_ARGS": "--model o3 --debug"},
            clear=False,
        ):
            assert _resolve_args() == ["--model", "o3", "--debug"]

    def test_resolve_command_prefers_env(self):
        with patch.dict(
            "os.environ",
            {"HERMES_CODEX_ACP_COMMAND": "/opt/codex/bin/codex-acp"},
            clear=False,
        ):
            client = CodexACPClient()
            assert client._acp_command == "/opt/codex/bin/codex-acp"

    def test_resolve_args_adapter_no_default_flags(self):
        assert _resolve_args("codex-acp") == []
        assert _resolve_args("/usr/local/bin/codex-acp") == []

    def test_resolve_args_from_credentials(self):
        with patch(
            "hermes_cli.auth.shutil.which",
            return_value="/usr/local/bin/codex-acp",
        ):
            with patch(
                "hermes_cli.auth._external_process_auth_present",
                return_value=None,
            ):
                creds = resolve_external_process_provider_credentials(
                    "codex-acp"
                )
                assert creds["provider"] == "codex-acp"
                assert creds["api_key"] == "codex-acp"
                assert creds["base_url"] == "acp://codex"
                assert creds["command"] == "/usr/local/bin/codex-acp"
                assert creds["args"] == []

    def test_subprocess_env_sets_codex_path_and_no_browser(self):
        with patch("shutil.which", return_value="/usr/local/bin/codex-acp"):
            with patch(
                "agent.codex_acp_client._resolve_codex_path",
                return_value="/usr/local/bin/codex.exe",
            ):
                client = CodexACPClient(command="/usr/local/bin/codex-acp")
                env = client._subprocess_env()
                assert env["NO_BROWSER"] == "1"
                assert env["CODEX_PATH"] == "/usr/local/bin/codex.exe"

    def test_spawn_argv_wraps_windows_batch_shims(self):
        with patch("os.name", "nt"):
            client = CodexACPClient(command="C:\\bin\\codex-acp.cmd", acp_args=[])
            argv = client._spawn_argv()
            assert argv == ["cmd", "/c", "C:\\bin\\codex-acp.cmd"]

    def test_spawn_argv_uses_resolved_argv_for_non_batch(self):
        with patch("os.name", "nt"):
            client = CodexACPClient(command="C:\\bin\\codex-acp.exe", acp_args=["--debug"])
            argv = client._spawn_argv()
            assert argv == ["C:\\bin\\codex-acp.exe", "--debug"]

    def test_authenticate_calls_api_key_method(self):
        with patch("agent.acp_client_base.BaseACPClient._rpc") as rpc:
            with patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "sk-test"},
                clear=False,
            ):
                client = CodexACPClient()
                client._authenticate(
                    {"authMethods": [{"id": "api-key"}]},
                    timeout_seconds=30.0,
                )
                assert rpc.call_count == 1
                args, _ = rpc.call_args
                assert args[0] == "authenticate"
                assert args[1]["methodId"] == "api-key"
