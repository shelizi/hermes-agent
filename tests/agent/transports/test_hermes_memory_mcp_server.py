"""Tests for the minimal ACP bridge to Hermes' built-in memory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def test_build_acp_server_config_is_memory_only():
    from agent.transports.hermes_memory_mcp_server import build_acp_server_config

    with (
        patch(
            "agent.transports.hermes_memory_mcp_server._memory_bridge_enabled",
            return_value=True,
        ),
        patch(
            "hermes_constants.get_hermes_home",
            return_value=Path("C:/hermes-test"),
        ),
    ):
        servers = build_acp_server_config()

    assert len(servers) == 1
    server = servers[0]
    assert server["name"] == "hermes-memory"
    assert Path(server["command"]).is_absolute()
    assert server["args"] == ["-m", "agent.transports.hermes_memory_mcp_server"]
    assert {item["name"] for item in server["env"]} >= {
        "HERMES_HOME",
        "PYTHONPATH",
    }


def test_memory_handler_persists_to_the_profile_store(tmp_path, monkeypatch):
    from agent.transports.hermes_memory_mcp_server import _memory
    from tools import write_approval

    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    monkeypatch.setattr(write_approval, "write_approval_enabled", lambda _: False)

    result = json.loads(
        _memory(
            action="add",
            target="user",
            content="User prefers concise Traditional Chinese replies.",
        )
    )

    assert result["success"] is True
    assert "User prefers concise Traditional Chinese replies." in (
        tmp_path / "USER.md"
    ).read_text(encoding="utf-8")


def test_memory_handler_reports_disabled_store(monkeypatch):
    from agent.transports.hermes_memory_mcp_server import _memory

    monkeypatch.setattr(
        "tools.memory_tool.load_on_disk_store",
        lambda: None,
    )
    result = json.loads(_memory(action="add", content="fact"))
    assert result["success"] is False
    assert "Memory is not available" in result["error"]
