"""Tests for the hermes-tools-as-MCP server module surface."""

from __future__ import annotations

import json
from pathlib import Path


class TestModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m

        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """Do not duplicate terminal/file tools already provided by Codex."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked

    def test_expected_hermes_specific_tools_listed(self):
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        for required in (
            "web_search",
            "web_extract",
            "browser_navigate",
            "vision_analyze",
            "image_generate",
            "skills_list",
            "skill_view",
            "skill_manage",
            "todo",
            "session_search",
        ):
            assert required in EXPOSED_TOOLS, f"missing {required!r}"

    def test_interactive_agent_loop_tools_not_exposed(self):
        """These require a live AIAgent or interactive UI callback."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        for unavailable in ("delegate_task", "clarify", "memory"):
            assert unavailable not in EXPOSED_TOOLS

    def test_kanban_worker_tools_exposed(self):
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        for worker_tool in (
            "kanban_complete",
            "kanban_block",
            "kanban_comment",
            "kanban_heartbeat",
        ):
            assert worker_tool in EXPOSED_TOOLS

    def test_kanban_orchestrator_tools_exposed(self):
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        for orch_tool in (
            "kanban_create",
            "kanban_show",
            "kanban_list",
            "kanban_unblock",
            "kanban_link",
        ):
            assert orch_tool in EXPOSED_TOOLS


class TestAcpBridge:
    def test_build_acp_server_config_preserves_the_session_allowlist(self, monkeypatch, tmp_path):
        from agent.transports.hermes_tools_mcp_server import build_acp_server_config

        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

        servers = build_acp_server_config(
            ["skills_list", "skill_view", "skill_manage", "todo", "session_search"]
        )

        assert len(servers) == 1
        server = servers[0]
        assert server["name"] == "hermes-tools"
        assert Path(server["command"]).is_absolute()
        assert server["args"] == ["-m", "agent.transports.hermes_tools_mcp_server"]
        env = {item["name"]: item["value"] for item in server["env"]}
        assert json.loads(env["HERMES_ACP_MCP_TOOLS"]) == [
            "session_search",
            "skill_manage",
            "skill_view",
            "skills_list",
            "todo",
        ]

    def test_stateless_todo_keeps_state_inside_the_mcp_process(self, monkeypatch):
        from agent.transports import hermes_tools_mcp_server as bridge

        monkeypatch.setattr(bridge, "_todo_store", None)

        first = json.loads(
            bridge._dispatch_stateless_tool(
                "todo",
                {"todos": [{"id": "one", "content": "First task", "status": "pending"}]},
            )
        )
        second = json.loads(bridge._dispatch_stateless_tool("todo", {}))

        assert first["todos"] == second["todos"]
        assert second["todos"][0]["id"] == "one"

    def test_stateless_dispatch_leaves_unrepresentable_tools_unhandled(self):
        from agent.transports.hermes_tools_mcp_server import _dispatch_stateless_tool

        assert _dispatch_stateless_tool("delegate_task", {}) is None
        assert _dispatch_stateless_tool("clarify", {}) is None


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1
