"""Hermes-tools-as-MCP server for external ACP runtimes.

Some ACP providers own their agent loop and build their own native tool list.
Without a bridge, Hermes' richer tool surface — web search, browser automation,
vision analysis, persistent memory, skills, cross-session search, image
generation, and TTS — is unreachable from those turns.

This module exposes a curated subset of Hermes tools to compatible ACP
subprocesses over stdio MCP. It is shared by native-MCP providers such as
Devin and Grok, and by the Codex app-server runtime when configured.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - browser_exec                         — Browser Use driver (explicit grant only)
  - execute_code                         — Hermes code runner (explicit grant only)
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list, skill_manage — Hermes' skill library
  - todo, session_search                 — stateless session-local helpers
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — provider-native shell/terminal
  - read_file / write_file / patch       — provider-native file/shell tools
  - search_files / process               — provider-native search/process tools
  - clarify                              — provider-native UX
  - delegate_task / clarify              — require the running AIAgent or
                                           interactive UI context and cannot
                                           be represented safely as a
                                           stateless MCP callback.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned/configured by ACP clients that support native MCP servers.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _resolve_mcp_server_class() -> Any:
    """Return the MCP server class across supported SDK layouts."""
    try:
        from mcp.server import MCPServer

        return MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP

            return FastMCP
        except ImportError:
            return None

# JSON Schema type -> Python type mapping for signature generation
_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _signature_from_schema(schema: dict | None) -> tuple[inspect.Signature, dict[str, type]]:
    """Synthesize a inspect.Signature and annotations dict from a JSON Schema.

    The mcp Python SDK generates MCP tool inputSchema by reflecting on the
    Python callable's signature and type annotations. Since Hermes tools
    are registered with JSON Schema parameter definitions (not typed Python
    callables), this helper creates a signature the SDK can reflect on.
    """
    if not schema or not isinstance(schema, dict):
        return inspect.Signature(), {}

    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    parameters = []
    annotations = {}

    for prop_name, prop_spec in props.items():
        if prop_name.startswith("_"):
            continue
        json_type = (prop_spec or {}).get("type", "string")
        py_type = _JSON_TO_PY.get(json_type, Any)

        if prop_name in required:
            param = inspect.Parameter(
                prop_name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=py_type,
            )
            annotations[prop_name] = py_type
        else:
            param = inspect.Parameter(
                prop_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[py_type],
            )
            annotations[prop_name] = Optional[py_type]

        parameters.append(param)

    sig = inspect.Signature(parameters, return_annotation=str)
    return sig, annotations


ACP_SERVER_NAME = "hermes-tools"
_ACP_ALLOWED_TOOLS_ENV = "HERMES_ACP_MCP_ALLOWED_TOOLS"
_todo_store = None

# Tools that require explicit allowlisting and must not be exposed by default.
_EXPLICIT_GRANT_ONLY_TOOLS = frozenset({"browser_exec", "execute_code"})

# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — native ACP providers already own those surfaces and their
#     approval flow should remain authoritative.
#   - delegate_task / clarify — these require a live AIAgent or interactive
#     UI callback and cannot be made correct by a stateless MCP process.
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "browser_exec",
    "execute_code",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    "skill_manage",
    "todo",
    "session_search",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running through an ACP
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)


def _apply_tool_schema(server: Any, name: str, parameters: dict[str, Any]) -> None:
    """Overwrite the reflected tool schema with Hermes' authoritative JSON schema.

    The mcp SDK's signature reflection loses enum values, field descriptions,
    default values, and nested object structures. Overwrite the generated
    Tool.parameters with the original JSON schema that Hermes sends the model,
    to retain descriptions, enums, defaults, and nested structures.
    """
    manager = getattr(server, "_tool_manager", None)
    tools = getattr(manager, "_tools", None)
    tool = tools.get(name) if isinstance(tools, dict) else None
    if tool is not None and hasattr(tool, "parameters"):
        tool.parameters = parameters


def build_acp_server_config(
    allowed_tools: Optional[list[str] | tuple[str, ...] | set[str]] = None,
) -> list[dict[str, Any]]:
    """Build the stdio MCP entry used by an external ACP provider.

    The provider receives only the Hermes tools already granted to the
    current session.  Keeping that allowlist in the child environment avoids
    accidentally exposing a broader process-global registry than the parent
    agent advertised.
    """
    if _resolve_mcp_server_class() is None:
        return []

    requested = (
        set(allowed_tools)
        if allowed_tools is not None
        else set(EXPOSED_TOOLS).difference(_EXPLICIT_GRANT_ONLY_TOOLS)
    )
    selected = sorted(requested.intersection(EXPOSED_TOOLS))
    if not selected:
        return []

    from hermes_constants import get_hermes_home

    source_root = str(Path(__file__).resolve().parents[2])
    python_path = os.environ.get("PYTHONPATH", "")
    if source_root not in python_path.split(os.pathsep):
        python_path = os.pathsep.join(part for part in (source_root, python_path) if part)

    return [{
        "name": ACP_SERVER_NAME,
        "command": str(Path(sys.executable).resolve()),
        "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
        "env": [
            {"name": "HERMES_HOME", "value": str(get_hermes_home())},
            {"name": "HERMES_QUIET", "value": "1"},
            {"name": "HERMES_REDACT_SECRETS", "value": "true"},
            {"name": "PYTHONPATH", "value": python_path},
            {"name": _ACP_ALLOWED_TOOLS_ENV, "value": json.dumps(selected)},
        ],
    }]


def _allowed_tools_from_env() -> set[str]:
    raw = os.environ.get(_ACP_ALLOWED_TOOLS_ENV, "").strip()
    if not raw:
        return set(EXPOSED_TOOLS).difference(_EXPLICIT_GRANT_ONLY_TOOLS)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    if not isinstance(value, list):
        return set()
    return {str(name) for name in value}.intersection(EXPOSED_TOOLS)


def _dispatch_stateless_tool(tool_name: str, kwargs: dict[str, Any]) -> Optional[str]:
    """Dispatch the ACP-safe tools that normally use AIAgent-owned state."""
    global _todo_store

    if tool_name == "session_search":
        from tools.session_search_tool import session_search

        return session_search(**kwargs)

    if tool_name == "todo":
        from tools.todo_tool import TodoStore, todo_tool

        if _todo_store is None:
            _todo_store = TodoStore()
        return todo_tool(
            todos=kwargs.get("todos"),
            merge=kwargs.get("merge", False),
            store=_todo_store,
        )

    return None


def _build_server() -> Any:
    """Create the MCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    _MCPServer = _resolve_mcp_server_class()
    if _MCPServer is None:  # pragma: no cover - install hint
        raise ImportError("hermes-tools MCP server requires the 'mcp' package")

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    mcp = _MCPServer(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside an external "
            "ACP session. Use these for capabilities the provider's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        ),
    )

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    exposed_count = 0

    allowed_tools = _allowed_tools_from_env()
    for name in EXPOSED_TOOLS:
        if name not in allowed_tools:
            continue
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        # The SDK wants a Python callable and derives the input schema from
        # its signature — there is no inputSchema parameter on either the
        # decorator or add_tool(). So build a closure that takes the arguments
        # dict, dispatches via handle_function_call, returns the result
        # string, and carries a __signature__ synthesized from the Hermes
        # JSON Schema (see _signature_from_schema) for the SDK to read.
        def _make_handler(tool_name: str, schema: dict | None):
            sig, annots = _signature_from_schema(schema)

            def _dispatch(**kwargs: Any) -> str:
                try:
                    args = {k: v for k, v in kwargs.items() if v is not None}
                    stateless_result = _dispatch_stateless_tool(tool_name, args)
                    if stateless_result is not None:
                        return stateless_result
                    return handle_function_call(tool_name, args)
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})

            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            _dispatch.__signature__ = sig
            _dispatch.__annotations__ = {**annots, "return": str}
            return _dispatch

        try:
            mcp.add_tool(
                _make_handler(name, params_schema),
                name=name,
                description=description,
            )
        except TypeError:
            # Older mcp SDK signature — fall back to decorator-style. The
            # synthesized __signature__ on the handler still drives schema
            # generation there.
            handler = _make_handler(name, params_schema)
            handler = mcp.tool(name=name, description=description)(handler)
        _apply_tool_schema(mcp, name, params_schema)

        exposed_count += 1

    logger.info(
        "hermes-tools MCP server registered %d/%d tools",
        exposed_count,
        len(EXPOSED_TOOLS),
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # MCPServer.run() defaults to stdio transport, which is what codex
    # spawns us on.
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
