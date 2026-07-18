"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation, vision analysis,
persistent memory, skills, cross-session search, image generation, TTS — is
unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list, skill_manage — Hermes' skill library
  - todo, session_search                 — stateless session-local helpers
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - delegate_task / clarify              — require the running AIAgent or
                                           interactive UI context and cannot
                                           be represented safely as a
                                           stateless MCP callback.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in.
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
    """Build a Python function signature and annotations from a JSON schema.

    Args:
        schema: JSON Schema dict with "properties" and "required" keys.

    Returns:
        (signature, annotations_dict) where signature has KEYWORD_ONLY params
        and annotations maps param names to Python types.
    """
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    params, annots = [], {}

    for pname, pspec in props.items():
        if pname.startswith("_"):
            continue
        py = _JSON_TO_PY.get((pspec or {}).get("type"), Any)
        ann, default = (
            (py, inspect.Parameter.empty)
            if pname in required
            else (Optional[py], None)
        )
        annots[pname] = ann
        params.append(
            inspect.Parameter(
                pname, inspect.Parameter.KEYWORD_ONLY, annotation=ann, default=default
            )
        )

    return inspect.Signature(params, return_annotation=str), annots


# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — codex's built-ins cover these and approval routes through
#     codex's own UI.
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
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)

ACP_SERVER_NAME = "hermes-tools"
_ACP_ALLOWED_TOOLS_ENV = "HERMES_ACP_MCP_TOOLS"
_todo_store: Any = None


def _apply_tool_schema(server: Any, name: str, parameters: dict[str, Any]) -> None:
    """Replace FastMCP's inferred schema with Hermes' authoritative schema.

    The synthetic signature is sufficient for older MCP SDKs.  Newer SDKs
    expose the registered Tool object, so also replace its published schema
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
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ImportError:
        return []

    requested = set(allowed_tools) if allowed_tools is not None else set(EXPOSED_TOOLS)
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
        return set(EXPOSED_TOOLS)
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
    """Create the FastMCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
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

        # FastMCP wants a Python callable. Build a closure that takes the
        # arguments dict, dispatches via handle_function_call, and returns
        # the result string. We use add_tool() for full control over the
        # input schema (FastMCP's @tool() decorator inspects type hints,
        # which we can't get from a JSON schema at runtime).
        def _make_handler(tool_name: str, schema: dict | None):
            sig, annots = _signature_from_schema(schema)

            def _dispatch(**kwargs: Any) -> str:
                try:
                    stateless_result = _dispatch_stateless_tool(tool_name, kwargs)
                    if stateless_result is not None:
                        return stateless_result
                    return handle_function_call(tool_name, kwargs or {})
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

    # FastMCP runs with stdio transport by default when launched as a
    # subprocess.
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
