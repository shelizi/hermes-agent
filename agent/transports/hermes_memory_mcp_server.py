"""Minimal MCP bridge for Hermes' built-in persistent memory.

ACP agents own their tool loop, so the normal Hermes ``memory`` agent-loop
dispatch is not available to them.  This module exposes only the built-in
file-backed memory tool through ACP's stdio MCP configuration.  It deliberately
does not expose the rest of Hermes' tool registry or any external provider.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal


SERVER_NAME = "hermes-memory"
SERVER_MODULE = "agent.transports.hermes_memory_mcp_server"


def _memory_bridge_enabled() -> bool:
    """Return whether either built-in memory store is enabled."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        memory_config = config.get("memory") or {}
        return bool(
            memory_config.get("memory_enabled")
            or memory_config.get("user_profile_enabled")
        )
    except Exception:
        return False


def build_acp_server_config() -> list[dict[str, Any]]:
    """Build the ACP ``session/new.mcpServers`` entry for built-in memory."""
    if not _memory_bridge_enabled():
        return []
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ImportError:
        # MCP is an optional Hermes extra.  Do not make an ACP session fail
        # merely because the optional bridge dependency is absent.
        return []

    from hermes_constants import get_hermes_home

    source_root = str(Path(__file__).resolve().parents[2])
    python_path = os.environ.get("PYTHONPATH", "")
    if source_root not in python_path.split(os.pathsep):
        python_path = os.pathsep.join(
            part for part in (source_root, python_path) if part
        )

    env = [
        {"name": "HERMES_HOME", "value": str(get_hermes_home())},
        {"name": "HERMES_QUIET", "value": "1"},
        {"name": "HERMES_REDACT_SECRETS", "value": "true"},
        {"name": "PYTHONPATH", "value": python_path},
    ]
    return [
        {
            "name": SERVER_NAME,
            "command": str(Path(sys.executable).resolve()),
            "args": ["-m", SERVER_MODULE],
            "env": env,
        }
    ]


def _memory(
    action: Literal["add", "replace", "remove"] | None = None,
    target: Literal["memory", "user"] = "memory",
    content: str | None = None,
    old_text: str | None = None,
    operations: list[dict[str, Any]] | None = None,
) -> str:
    """Read or mutate Hermes' built-in persistent memory on disk."""
    try:
        from tools.memory_tool import load_on_disk_store, memory_tool

        return memory_tool(
            action=action,
            target=target,
            content=content,
            old_text=old_text,
            operations=operations,
            store=load_on_disk_store(),
        )
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Memory bridge failed: {exc}"},
            ensure_ascii=False,
        )


def _build_server() -> Any:
    """Create the stdio MCP server lazily so importing this module is cheap."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            f"Hermes memory MCP bridge requires the 'mcp' package: {exc}"
        ) from exc

    from tools.memory_tool import MEMORY_SCHEMA

    server = FastMCP(SERVER_NAME)
    server.add_tool(
        _memory,
        name=MEMORY_SCHEMA["name"],
        description=MEMORY_SCHEMA["description"],
    )
    return server


def main(argv: list[str] | None = None) -> int:
    """Run the stdio MCP bridge."""
    del argv
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")
    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        sys.stderr.write(f"Hermes memory MCP bridge error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
