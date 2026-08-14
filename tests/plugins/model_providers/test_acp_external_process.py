"""Tests for ACP external-process provider hooks.

Drives the refactor of command-path and argument resolution from
hermes_cli/auth.py into ProviderProfile so new ACP providers can ship
as plugins without touching the core.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_providers(monkeypatch, tmp_path):
    """Force a fresh provider discovery using bundled plugins only."""
    import sys

    import hermes_constants
    import providers

    token = hermes_constants.set_hermes_home_override(str(tmp_path))
    monkeypatch.setattr(providers, "_discovered", False)
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    for mod in list(sys.modules.keys()):
        if mod.startswith("plugins.model_providers.") or mod.startswith(
            "_hermes_user_provider_"
        ):
            sys.modules.pop(mod, None)
    yield
    hermes_constants.reset_hermes_home_override(token)


@pytest.fixture
def devin_profile(isolated_providers):
    import providers

    profile = providers.get_provider_profile("devin-acp")
    assert profile is not None
    return profile


@pytest.fixture
def grok_profile(isolated_providers):
    import providers

    profile = providers.get_provider_profile("grok-acp")
    assert profile is not None
    return profile


@pytest.fixture
def codex_profile(isolated_providers):
    import providers

    profile = providers.get_provider_profile("codex-acp")
    assert profile is not None
    return profile


@pytest.fixture
def antigravity_profile(isolated_providers):
    import providers

    profile = providers.get_provider_profile("antigravity-acp")
    assert profile is not None
    return profile


@pytest.fixture
def copilot_profile(isolated_providers):
    import providers

    profile = providers.get_provider_profile("copilot-acp")
    assert profile is not None
    return profile


class TestProviderProfileSearchCommandPath:
    """ProviderProfile.search_command_path finds known install locations."""

    def test_default_returns_none(self, copilot_profile):
        assert copilot_profile.search_command_path("copilot") is None

    def test_devin_finds_windows_installer_path(self, devin_profile, monkeypatch, tmp_path):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("pathlib.Path.is_file", return_value=True):
                path = devin_profile.search_command_path("devin")
        assert path is not None
        assert path.casefold().endswith(r"devin\cli\bin\devin.exe".casefold())

    def test_grok_finds_home_dot_grok_bin(self, grok_profile, monkeypatch, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("pathlib.Path.is_file", return_value=True):
                path = grok_profile.search_command_path("grok")
        assert path is not None
        assert ".grok/bin/grok" in Path(path).as_posix().casefold()

    def test_codex_finds_npm_prefix(self, codex_profile, monkeypatch, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("pathlib.Path.is_file", return_value=True):
                path = codex_profile.search_command_path("codex-acp")
        assert path is not None
        assert ".hermes/codex-acp/node_modules/.bin/codex-acp" in Path(path).as_posix().casefold()

    def test_antigravity_finds_home_dot_antigravity_bin(
        self, antigravity_profile, monkeypatch, tmp_path
    ):
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("pathlib.Path.is_file", return_value=True):
                path = antigravity_profile.search_command_path("agy")
        assert path is not None
        assert ".antigravity/bin/agy" in Path(path).as_posix().casefold()


class TestProviderProfileResolveCommandArgs:
    """ProviderProfile.resolve_command_args filters default launch args."""

    def test_default_returns_default_args_unchanged(self, copilot_profile):
        assert copilot_profile.resolve_command_args(
            "copilot", ["--acp", "--stdio"]
        ) == ["--acp", "--stdio"]

    def test_devin_default_returns_default_args_unchanged(self, devin_profile):
        assert devin_profile.resolve_command_args("devin", ["acp"]) == ["acp"]

    def test_antigravity_native_gets_default_args(self, antigravity_profile):
        assert antigravity_profile.resolve_command_args("agy", ["--acp"]) == ["--acp"]
        assert antigravity_profile.resolve_command_args(
            r"C:\Program Files\antigravity\bin\agy.exe", ["--acp"]
        ) == ["--acp"]

    def test_antigravity_adapter_gets_empty_args(self, antigravity_profile):
        assert antigravity_profile.resolve_command_args("agy-acp", ["--acp"]) == []
        assert antigravity_profile.resolve_command_args(
            "/usr/local/bin/antigravity-acp", ["--acp"]
        ) == []


class TestAuthUsesProviderProfileHooks:
    """hermes_cli/auth.py delegates to ProviderProfile hooks."""

    def test_resolve_command_path_calls_profile_search_when_not_on_path(
        self, isolated_providers, monkeypatch
    ):
        import hermes_cli.auth as auth
        import providers

        monkeypatch.setattr(auth.shutil, "which", lambda _cmd: None)

        devin = providers.get_provider_profile("devin-acp")
        assert devin is not None
        monkeypatch.setattr(devin, "search_command_path", lambda _cmd: r"C:\sentinel\devin.exe")

        path = auth._resolve_external_process_command_path("devin-acp", "devin")
        assert path == r"C:\sentinel\devin.exe"

    def test_resolve_command_args_calls_profile_resolve(
        self, isolated_providers, monkeypatch
    ):
        import hermes_cli.auth as auth
        import providers

        monkeypatch.setenv("HERMES_ANTIGRAVITY_ACP_COMMAND", "agy-acp")

        antigravity = providers.get_provider_profile("antigravity-acp")
        assert antigravity is not None
        monkeypatch.setattr(
            antigravity, "resolve_command_args", lambda _cmd, _default: ["__sentinel__"]
        )

        command, args = auth._resolve_external_process_command_args("antigravity-acp")
        assert command == "agy-acp"
        assert args == ["__sentinel__"]


_DEVIN_SAMPLE_ERR = (
    "Error: Unknown model: '__hermes_devin_probe__'\n"
    "Available: adaptive, claude-opus-4.8, swe-1.7, gpt-5.5\n"
)

_GROK_SAMPLE_OUT = (
    "Available models:\n"
    "  * grok-4.5 (default)\n"
    "  - grok-composer-2.5-fast\n"
    "  - grok-build-0.1\n"
)


class TestProviderProfileFetchModels:
    """ProviderProfile.fetch_models discovers ACP model lists from the CLI."""

    def test_devin_profile_fetch_models_runs_cli_probe(self, devin_profile, monkeypatch):
        mock_proc = MagicMock()
        mock_proc.stderr = _DEVIN_SAMPLE_ERR
        mock_proc.stdout = ""

        with patch("subprocess.run", return_value=mock_proc) as run:
            with patch("shutil.which", return_value="/bin/devin"):
                models = devin_profile.fetch_models(timeout=1.0)

        assert models == ["adaptive", "claude-opus-4.8", "swe-1.7", "gpt-5.5"]
        argv = run.call_args[0][0]
        assert argv[0] == "/bin/devin"
        assert "--model" in argv
        assert "-p" in argv

    def test_devin_profile_fetch_models_returns_empty_on_timeout(self, devin_profile, monkeypatch):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("devin", 1)
        ):
            with patch("shutil.which", return_value="/bin/devin"):
                models = devin_profile.fetch_models(timeout=1.0)
        assert models == []

    def test_grok_profile_fetch_models_runs_models_subcommand(self, grok_profile, monkeypatch):
        mock_proc = MagicMock()
        mock_proc.stderr = ""
        mock_proc.stdout = _GROK_SAMPLE_OUT

        with patch("subprocess.run", return_value=mock_proc) as run:
            with patch("shutil.which", return_value="/bin/grok"):
                models = grok_profile.fetch_models(timeout=1.0)

        assert models == ["grok-4.5", "grok-composer-2.5-fast", "grok-build-0.1"]
        argv = run.call_args[0][0]
        assert argv[0] == "/bin/grok"
        assert "models" in argv

    def test_grok_profile_fetch_models_returns_empty_on_timeout(self, grok_profile, monkeypatch):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("grok", 1)
        ):
            with patch("shutil.which", return_value="/bin/grok"):
                models = grok_profile.fetch_models(timeout=1.0)
        assert models == []

    def test_provider_model_ids_uses_external_process_profile_fetch_models(
        self, isolated_providers, monkeypatch
    ):
        import hermes_cli.models as models
        import providers

        devin = providers.get_provider_profile("devin-acp")
        assert devin is not None
        monkeypatch.setattr(devin, "search_command_path", lambda _cmd: None)
        monkeypatch.setattr(devin, "fetch_models", lambda **_: ["__live_devin__"])

        with patch("shutil.which", return_value=None), patch(
            "subprocess.run", side_effect=FileNotFoundError
        ):
            ids = models.provider_model_ids("devin-acp", force_refresh=True)
        assert ids == ["__live_devin__"]

    def test_provider_model_ids_falls_back_to_curated_when_fetch_models_empty(
        self, isolated_providers, monkeypatch
    ):
        import hermes_cli.models as models
        import providers

        grok = providers.get_provider_profile("grok-acp")
        assert grok is not None
        monkeypatch.setattr(grok, "search_command_path", lambda _cmd: None)
        monkeypatch.setattr(grok, "fetch_models", lambda **_: [])

        with patch("shutil.which", return_value=None), patch(
            "subprocess.run", side_effect=FileNotFoundError
        ):
            ids = models.provider_model_ids("grok-acp", force_refresh=True)
        assert "grok-4.5" in ids
        assert "grok-acp" in ids
