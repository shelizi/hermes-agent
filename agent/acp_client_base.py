"""Generic OpenAI-compatible ACP client base.

Provides the shared lifecycle, JSON-RPC transport, message formatting,
and stream conversion used by provider-specific ACP clients.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import pathname2url

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, get_write_denied_error, is_write_approval_required
from agent.redact import redact_sensitive_text
from hermes_cli._subprocess_compat import windows_hide_flags
from tools.ansi_strip import sanitize_display_text
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 900.0
_REUSE_DISABLE_VALUES = frozenset({"0", "false", "no", "off"})

# Outbound image payload cap (decoded bytes). Keeps a single session/prompt
# from shipping multi-hundred-MB base64 blobs into an ACP child process.
_MAX_ACP_IMAGE_BYTES = 15 * 1024 * 1024

# OpenAI / Responses-style multimodal part types Hermes uses for images.
_OPENAI_IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})
_OPENAI_AUDIO_PART_TYPES = frozenset({"audio", "input_audio"})
_OPENAI_FILE_PART_TYPES = frozenset({"file", "input_file", "document"})

# A few ACP implementations write the JSON-RPC response before the final
# session/update notification has made it through their event loop.  The
# normal post-response quiet window is intentionally short, but an empty
# response needs a little more time for the first delayed agent chunk.  This
# is a transport-level compatibility grace period, not a model/API timeout.
_ACP_POST_RESPONSE_IDLE_SECONDS = 0.15
_ACP_POST_RESPONSE_STABLE_CHECKS = 2
_ACP_FIRST_CHUNK_GRACE_SECONDS = 2.0


def _acp_process_reuse_enabled() -> bool:
    raw = os.getenv("HERMES_ACP_PROCESS_REUSE", "1").strip().lower()
    return raw not in _REUSE_DISABLE_VALUES


def _acp_session_reuse_enabled() -> bool:
    """Session continuity defaults to on whenever process reuse is on."""
    raw = os.getenv("HERMES_ACP_SESSION_REUSE", "").strip().lower()
    if raw in _REUSE_DISABLE_VALUES:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _acp_process_reuse_enabled()


_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call\b[^>]*>\s*(.*?)\s*</tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call\b[^>]*>", re.IGNORECASE)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)


def _marker_prefix_suffix_length(text: str, marker: str) -> int:
    """Length of the longest suffix of ``text`` that prefixes ``marker``."""
    text_lower = text.lower()
    marker_lower = marker.lower()
    for size in range(min(len(text), len(marker) - 1), 0, -1):
        if text_lower.endswith(marker_lower[:size]):
            return size
    return 0


class _ToolCallStreamFilter:
    """Suppress textual ``<tool_call>`` blocks without delaying normal text.

    ACP providers stream arbitrary chunk boundaries, so both the opening and
    closing tags may be split across updates. The filter retains only the
    short suffix needed to recognize a split marker and discards tool payload
    bytes as they arrive. The completed response is parsed separately into
    structured OpenAI/Hermes ``tool_calls``.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._inside_tool_call = False

    def feed(self, chunk: str) -> str:
        if not isinstance(chunk, str) or not chunk:
            return ""
        self._pending += chunk
        visible: list[str] = []

        while self._pending:
            pending_lower = self._pending.lower()
            if self._inside_tool_call:
                close_at = pending_lower.find(_TOOL_CALL_CLOSE)
                if close_at >= 0:
                    self._pending = self._pending[close_at + len(_TOOL_CALL_CLOSE):]
                    self._inside_tool_call = False
                    continue

                keep = _marker_prefix_suffix_length(self._pending, _TOOL_CALL_CLOSE)
                self._pending = self._pending[-keep:] if keep else ""
                break

            open_at = pending_lower.find(_TOOL_CALL_OPEN)
            if open_at >= 0:
                if open_at:
                    visible.append(self._pending[:open_at])
                self._pending = self._pending[open_at + len(_TOOL_CALL_OPEN):]
                self._inside_tool_call = True
                continue

            keep = _marker_prefix_suffix_length(self._pending, _TOOL_CALL_OPEN)
            if keep:
                visible.append(self._pending[:-keep])
                self._pending = self._pending[-keep:]
            else:
                visible.append(self._pending)
                self._pending = ""
            break

        return "".join(visible)

    def finish(self) -> str:
        """Flush ordinary pending text; discard an unterminated tool block."""
        if self._inside_tool_call:
            self._pending = ""
            return ""
        tail = self._pending
        self._pending = ""
        return tail


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    # Copilot ACP is a model-driving CLI executor: it legitimately needs LLM
    # provider credentials. Route through the central helper so Tier-1 secrets
    # (gateway bot tokens, GitHub auth, infra) are still stripped (#29157).
    env = hermes_subprocess_env(inherit_credentials=True)
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _acp_rpc_error_message(error: Any) -> str:
    """Extract a safe, actionable message from an ACP JSON-RPC error.

    ACP agents commonly wrap the upstream provider failure in
    ``error.data.message`` while leaving ``error.message`` as the generic
    ``Internal error``.  Prefer that nested detail, but redact it before it
    can reach logs or a user-facing response.
    """
    if isinstance(error, dict):
        outer_message = str(error.get("message") or "").strip()
        data = error.get("data")
        detail = ""
        status = None
        if isinstance(data, dict):
            detail = str(data.get("message") or "").strip()
            status = data.get("http_status") or data.get("status") or data.get("status_code")
        elif isinstance(data, str):
            detail = data.strip()

        if detail:
            status_text = str(status or "").strip()
            if status_text and status_text not in detail:
                detail = f"HTTP {status_text}: {detail}"
            return redact_sensitive_text(detail, force=True)
        if outer_message:
            return redact_sensitive_text(outer_message, force=True)

    return redact_sensitive_text(str(error), force=True)


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    }


def _permission_auto_selected(message_id: Any, options: Any) -> dict[str, Any]:
    """Auto-select an allow option so ACP agent tools can run without a UI.

    Preference: allow_always → allow_once → first option. Falls back to
    cancelled when no options are present.
    """
    opts = options if isinstance(options, list) else []
    option_id = None
    for preferred in ("allow_always", "allow_once"):
        for opt in opts:
            if not isinstance(opt, dict):
                continue
            kind = str(opt.get("kind") or "").strip().lower()
            if kind == preferred:
                option_id = opt.get("optionId") or opt.get("option_id")
                if option_id:
                    break
        if option_id:
            break
    if not option_id:
        for opt in opts:
            if not isinstance(opt, dict):
                continue
            kind = str(opt.get("kind") or "").strip().lower()
            if kind.startswith("allow"):
                option_id = opt.get("optionId") or opt.get("option_id")
                if option_id:
                    break
    if not option_id and opts and isinstance(opts[0], dict):
        option_id = opts[0].get("optionId") or opts[0].get("option_id")
    if not option_id:
        return _permission_denied(message_id)
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "selected",
                "optionId": str(option_id),
            }
        },
    }


def _acp_auto_approve_enabled() -> bool:
    """Whether ACP permission prompts should be auto-approved.

    Default on: ACP backends (Devin/Copilot) own their tool loop and Hermes
    has no interactive permission UI in the chat path. Set
    ``HERMES_ACP_AUTO_APPROVE=0`` to restore deny-all.
    """
    raw = os.getenv("HERMES_ACP_AUTO_APPROVE", "1").strip().lower()
    return raw not in _REUSE_DISABLE_VALUES


def _tool_update_text_preview(update: dict[str, Any], *, limit: int = 240) -> str:
    """Best-effort human preview from an ACP tool_call / tool_call_update."""
    title = str(update.get("title") or "").strip()
    chunks: list[str] = []
    content = update.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            # type: content → nested content.text; or plain text fields
            inner = block.get("content")
            if isinstance(inner, dict):
                text = str(inner.get("text") or "").strip()
                if text:
                    chunks.append(text)
            text = str(block.get("text") or "").strip()
            if text:
                chunks.append(text)
            if block.get("type") == "diff":
                path = str(block.get("path") or "").strip()
                if path:
                    chunks.append(f"diff {path}")
    elif isinstance(content, dict):
        text = str(content.get("text") or "").strip()
        if text:
            chunks.append(text)
    raw_in = update.get("rawInput")
    if isinstance(raw_in, dict) and not chunks:
        # compact path/command hints
        for key in ("path", "file", "command", "query", "url", "pattern"):
            if raw_in.get(key):
                chunks.append(f"{key}={raw_in.get(key)}")
                break
    body = " · ".join(chunks).strip()
    if title and body:
        preview = f"{title}: {body}"
    else:
        preview = title or body or "ACP tool"
    # ACP tool content is subprocess/terminal output.  Strip ECMA-48 styling
    # and control bytes before it reaches a Desktop status strip or an opted-in
    # gateway progress bubble (PowerShell commonly emits these around tables).
    preview = sanitize_display_text(preview).strip()
    if len(preview) > limit:
        return preview[: limit - 1] + "…"
    return preview


def _tool_kind_name(update: dict[str, Any]) -> str:
    kind = str(update.get("kind") or "other").strip().lower() or "other"
    # Prefer a stable synthetic name the Desktop tool strip can show.
    return f"acp_{kind}"


def _message_continuity_key(message: dict[str, Any]) -> tuple[Any, ...]:
    """Stable identity for prefix-matching conversation history."""
    role = str(message.get("role") or "").strip().lower()
    content = _render_message_content(message.get("content"))
    tool_sig: tuple[Any, ...] = ()
    raw_tcs = message.get("tool_calls")
    if isinstance(raw_tcs, list) and raw_tcs:
        names: list[str] = []
        for tc in raw_tcs:
            if isinstance(tc, dict):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                names.append(str(fn.get("name") or tc.get("name") or ""))
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else getattr(tc, "name", None)
                names.append(str(name or ""))
        tool_sig = tuple(names)
    tool_call_id = str(message.get("tool_call_id") or "")
    return (role, content, tool_sig, tool_call_id)


def _common_message_prefix_len(
    previous: list[dict[str, Any]] | None,
    current: list[dict[str, Any]],
) -> int:
    if not previous:
        return 0
    n = 0
    for left, right in zip(previous, current):
        if not isinstance(left, dict) or not isinstance(right, dict):
            break
        if _message_continuity_key(left) != _message_continuity_key(right):
            break
        n += 1
    return n


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    *,
    continuation: bool = False,
) -> str:
    sections: list[str] = []
    if continuation:
        sections.append(
            "Continue the same ACP session. The messages below are NEW since "
            "the previous prompt — do not restate earlier context unless needed."
        )
    else:
        sections.extend(
            [
                "You are being used as the active ACP agent backend for Hermes.",
                "Use ACP capabilities to complete tasks.",
                "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
                "If no tool is needed, answer normally.",
            ]
        )
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        # Keep tool-call-only assistant turns in the transcript for continuity.
        if not rendered and role == "assistant" and message.get("tool_calls"):
            rendered = "[tool_calls]"
        if not rendered and role == "tool":
            rendered = "[tool result]"
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        heading = "New messages:\n\n" if continuation else "Conversation transcript:\n\n"
        sections.append(heading + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    continue
                # Keep multimodal parts visible in the transcript as compact
                # placeholders — actual pixels/files travel as ACP content
                # blocks on session/prompt when the agent advertises support.
                placeholder = _media_placeholder_for_part(item)
                if placeholder:
                    parts.append(placeholder)
        return "\n".join(parts).strip()
    return str(content).strip()


def _media_placeholder_for_part(part: dict[str, Any]) -> str:
    """Return a short transcript placeholder for a multimodal content part."""
    ptype = str(part.get("type") or "").strip().lower()
    if ptype in _OPENAI_IMAGE_PART_TYPES:
        mime = _guess_image_mime_from_part(part) or "image"
        return f"[Image attached ({mime})]"
    if ptype in _OPENAI_AUDIO_PART_TYPES:
        mime = str(part.get("mimeType") or part.get("mime_type") or "audio").strip()
        return f"[Audio attached ({mime})]"
    if ptype in _OPENAI_FILE_PART_TYPES:
        name = (
            str(part.get("name") or part.get("filename") or "").strip()
            or str((part.get("file") or {}).get("filename") or "").strip()
            or "file"
        )
        return f"[File attached: {name}]"
    return ""


def _guess_image_mime_from_part(part: dict[str, Any]) -> str | None:
    for key in ("mimeType", "mime_type", "media_type"):
        raw = part.get(key)
        if isinstance(raw, str) and raw.strip().startswith("image/"):
            return raw.strip().lower()
    url = _image_url_from_part(part)
    if url.startswith("data:"):
        header = url.split(",", 1)[0]
        mime = header[len("data:") :].split(";", 1)[0].strip().lower()
        if mime.startswith("image/"):
            return mime
    if url:
        guessed, _ = mimetypes.guess_type(url.split("?", 1)[0])
        if guessed and guessed.startswith("image/"):
            return guessed
    return None


def _image_url_from_part(part: dict[str, Any]) -> str:
    """Pull a URL / data-URL / path string out of an OpenAI-style image part."""
    ptype = str(part.get("type") or "").strip().lower()
    if ptype == "image":
        # Anthropic-ish or already-ACP-shaped
        data = part.get("data")
        mime = str(part.get("mimeType") or part.get("mime_type") or "image/png").strip()
        if isinstance(data, str) and data.strip():
            if data.startswith("data:"):
                return data.strip()
            return f"data:{mime};base64,{data.strip()}"
        uri = part.get("uri") or part.get("url") or ""
        return str(uri or "").strip()

    image_value = part.get("image_url", part.get("image", part.get("input_image")))
    if isinstance(image_value, dict):
        url = image_value.get("url") or image_value.get("uri") or ""
        return str(url or "").strip()
    if isinstance(image_value, str):
        return image_value.strip()
    # Bare url field on some shims
    url = part.get("url") or part.get("uri") or ""
    return str(url or "").strip()


def _path_from_file_uri_or_local(url: str) -> Path | None:
    """Resolve file:// URIs and bare local paths to an existing Path."""
    raw = (url or "").strip()
    if not raw:
        return None
    if raw.startswith("file:"):
        parsed = urlparse(raw)
        path_str = unquote(parsed.path or "")
        # Windows file:///C:/... → path starts with /C:/
        if os.name == "nt" and path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
            path_str = path_str[1:]
        candidate = Path(path_str)
    else:
        # data: / http(s): are not local files
        if "://" in raw:
            return None
        candidate = Path(os.path.expanduser(raw))
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


def _openai_image_part_to_acp_block(
    part: dict[str, Any],
    *,
    max_bytes: int = _MAX_ACP_IMAGE_BYTES,
) -> dict[str, Any] | None:
    """Convert an OpenAI multimodal image part into an ACP ``image`` ContentBlock.

    Returns None when the payload cannot be materialised as base64 within
    *max_bytes* (missing data, oversized, unreadable path).
    """
    url = _image_url_from_part(part)
    if not url:
        return None

    mime = _guess_image_mime_from_part(part) or "image/png"
    data_b64: str | None = None
    uri: str | None = None

    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if not payload:
            return None
        # Some producers include whitespace/newlines in base64 payloads.
        data_b64 = "".join(payload.split())
        mime_part = header[len("data:") :].split(";", 1)[0].strip().lower()
        if mime_part.startswith("image/"):
            mime = mime_part
        try:
            raw_len = len(base64.b64decode(data_b64, validate=False))
        except Exception:
            return None
        if raw_len > max_bytes:
            logger.warning(
                "ACP image data-URL too large (%d bytes > %d); skipping",
                raw_len,
                max_bytes,
            )
            return None
        uri = None
    else:
        path = _path_from_file_uri_or_local(url)
        if path is not None:
            try:
                size = path.stat().st_size
                if size > max_bytes:
                    logger.warning(
                        "ACP image file too large (%s: %d bytes > %d); skipping",
                        path,
                        size,
                        max_bytes,
                    )
                    return None
                raw = path.read_bytes()
            except OSError as exc:
                logger.warning("ACP image file unreadable (%s): %s", path, exc)
                return None
            data_b64 = base64.b64encode(raw).decode("ascii")
            guessed, _ = mimetypes.guess_type(str(path))
            if guessed and guessed.startswith("image/"):
                mime = guessed
            try:
                uri = path.resolve().as_uri()
            except Exception:
                uri = f"file://{pathname2url(str(path.resolve()))}"
        elif url.startswith(("http://", "https://")):
            # Protocol requires base64 `data`; remote URLs alone are not enough
            # without a fetch. Prefer resource_link so the agent can pull them.
            return None
        else:
            return None

    if not data_b64:
        return None

    block: dict[str, Any] = {
        "type": "image",
        "mimeType": mime,
        "data": data_b64,
    }
    if uri:
        block["uri"] = uri
    return block


def _openai_file_part_to_resource_link(part: dict[str, Any]) -> dict[str, Any] | None:
    """Map a file/document part onto an ACP ``resource_link`` (always allowed)."""
    file_obj = part.get("file") if isinstance(part.get("file"), dict) else {}
    name = (
        str(part.get("name") or part.get("filename") or "").strip()
        or str(file_obj.get("filename") or file_obj.get("name") or "").strip()
        or "attachment"
    )
    mime = (
        str(part.get("mimeType") or part.get("mime_type") or "").strip()
        or str(file_obj.get("mimeType") or file_obj.get("mime_type") or "").strip()
        or None
    )
    url = (
        str(part.get("url") or part.get("uri") or part.get("path") or "").strip()
        or str(file_obj.get("url") or file_obj.get("uri") or file_obj.get("path") or "").strip()
    )
    if not url:
        # OpenAI file_id-only parts have no client-side path we can share.
        return None

    path = _path_from_file_uri_or_local(url)
    if path is not None:
        try:
            uri = path.resolve().as_uri()
            size = path.stat().st_size
        except OSError:
            return None
        if not mime:
            guessed, _ = mimetypes.guess_type(str(path))
            mime = guessed
        block: dict[str, Any] = {
            "type": "resource_link",
            "uri": uri,
            "name": name if name != "attachment" else path.name,
        }
        if mime:
            block["mimeType"] = mime
        if size is not None:
            block["size"] = int(size)
        return block

    if url.startswith(("http://", "https://", "file:")):
        block = {
            "type": "resource_link",
            "uri": url,
            "name": name,
        }
        if mime:
            block["mimeType"] = mime
        return block
    return None


def _extract_acp_media_blocks(
    messages: list[dict[str, Any]],
    *,
    max_image_bytes: int = _MAX_ACP_IMAGE_BYTES,
) -> list[dict[str, Any]]:
    """Collect ACP ContentBlocks (image / resource_link) from Hermes messages.

    Deduplicates by (type, data-or-uri) so continuation turns that re-send
    history-adjacent images don't inflate the prompt.
    """
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(block: dict[str, Any] | None) -> None:
        if not block:
            return
        key = f"{block.get('type')}|{block.get('data') or ''}|{block.get('uri') or ''}"
        # Truncate key for huge base64 to avoid giant set entries
        if len(key) > 200:
            key = f"{key[:80]}…{key[-40:]}|{len(key)}"
        if key in seen:
            return
        seen.add(key)
        blocks.append(block)

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            ptype = str(item.get("type") or "").strip().lower()
            if ptype in _OPENAI_IMAGE_PART_TYPES:
                image_block = _openai_image_part_to_acp_block(
                    item, max_bytes=max_image_bytes
                )
                if image_block:
                    _add(image_block)
                else:
                    # Fall back: if we only have a remote URL, surface as link.
                    url = _image_url_from_part(item)
                    if url.startswith(("http://", "https://")):
                        _add(
                            {
                                "type": "resource_link",
                                "uri": url,
                                "name": Path(urlparse(url).path).name or "image",
                                "mimeType": _guess_image_mime_from_part(item)
                                or "image/*",
                            }
                        )
            elif ptype in _OPENAI_FILE_PART_TYPES:
                _add(_openai_file_part_to_resource_link(item))
            elif ptype == "resource_link":
                # Already ACP-shaped (e.g. round-tripped through Hermes).
                uri = str(item.get("uri") or "").strip()
                name = str(item.get("name") or "resource").strip() or "resource"
                if uri:
                    block = {
                        "type": "resource_link",
                        "uri": uri,
                        "name": name,
                    }
                    mime = str(item.get("mimeType") or item.get("mime_type") or "").strip()
                    if mime:
                        block["mimeType"] = mime
                    size = item.get("size")
                    if isinstance(size, int):
                        block["size"] = size
                    _add(block)
    return blocks


def _parse_prompt_capabilities(init_result: dict[str, Any] | None) -> dict[str, bool]:
    """Extract agent promptCapabilities from an ACP initialize result."""
    caps = {"image": False, "audio": False, "embeddedContext": False}
    if not isinstance(init_result, dict):
        return caps
    agent_caps = init_result.get("agentCapabilities") or init_result.get(
        "agent_capabilities"
    )
    if not isinstance(agent_caps, dict):
        return caps
    prompt_caps = agent_caps.get("promptCapabilities") or agent_caps.get(
        "prompt_capabilities"
    )
    if not isinstance(prompt_caps, dict):
        return caps
    for key in ("image", "audio", "embeddedContext"):
        # Also accept snake_case from some SDKs
        alt = "embedded_context" if key == "embeddedContext" else key
        val = prompt_caps.get(key)
        if val is None:
            val = prompt_caps.get(alt)
        caps[key] = bool(val)
    return caps


def _build_acp_prompt_blocks(
    prompt_text: str,
    media_blocks: list[dict[str, Any]] | None,
    *,
    prompt_capabilities: dict[str, bool] | None,
) -> list[dict[str, Any]]:
    """Assemble the ``session/prompt`` ``prompt`` array (text + optional media).

    Per ACP, text + resource_link are always allowed; image/audio/resource
    require the matching ``promptCapabilities`` flag from initialize.
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompt_text,
        }
    ]
    caps = prompt_capabilities or {}
    allow_image = bool(caps.get("image"))
    allow_audio = bool(caps.get("audio"))
    allow_embedded = bool(caps.get("embeddedContext"))

    skipped: list[str] = []
    for media in media_blocks or []:
        if not isinstance(media, dict):
            continue
        mtype = str(media.get("type") or "").strip().lower()
        if mtype == "text":
            text = str(media.get("text") or "").strip()
            if text:
                blocks.append({"type": "text", "text": text})
        elif mtype == "image":
            if allow_image:
                blocks.append(media)
            else:
                skipped.append("image")
        elif mtype == "audio":
            if allow_audio:
                blocks.append(media)
            else:
                skipped.append("audio")
        elif mtype == "resource":
            if allow_embedded:
                blocks.append(media)
            else:
                skipped.append("resource")
        elif mtype == "resource_link":
            # Baseline: all Agents MUST support ResourceLink.
            blocks.append(media)
        else:
            logger.debug("Ignoring unknown ACP media block type %r", mtype)

    if skipped:
        # Note once in the text block so the model knows something was dropped.
        note = (
            f"\n\n[Hermes note: {', '.join(sorted(set(skipped)))} attachment(s) "
            "were not forwarded because this ACP agent did not advertise the "
            "matching promptCapabilities flag.]"
        )
        blocks[0] = {
            "type": "text",
            "text": str(blocks[0].get("text") or "") + note,
        }
    return blocks


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


def _tool_names_from_schemas(
    tools: list[dict[str, Any]] | None,
) -> set[str] | None:
    """Return the exact tool-name allowlist, or None when unrestricted."""
    if tools is None:
        return None
    names: set[str] = set()
    for item in tools:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _extract_tool_calls_from_text(
    text: str,
    *,
    allowed_tool_names: set[str] | None = None,
) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return

        # OpenAI nested shape and Devin SWE 1.7 compact shape.
        fn = obj.get("function")
        if isinstance(fn, dict):
            fn_name = fn.get("name")
            fn_args = fn.get("arguments", "{}")
        else:
            fn_name = obj.get("name")
            fn_args = obj.get("arguments", {})

        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_name = fn_name.strip()
        if allowed_tool_names is not None and fn_name not in allowed_tool_names:
            logger.warning(
                "ACP textual tool call ignored because tool is not allowed: %s",
                fn_name,
            )
            return

        if fn_args is None:
            fn_args = {}
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=fn_name,
                arguments=fn_args,
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add_tool_call(m.group(1))
        consumed_spans.append((m.start(), m.end()))

    # Only try bare OpenAI JSON fallback when no XML tool blocks were found.
    if not consumed_spans:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add_tool_call(m.group(0))
            consumed_spans.append((m.start(), m.end()))

    # Suppress a truncated model control block through end-of-text.
    for m in _TOOL_CALL_OPEN_RE.finditer(text):
        if any(start <= m.start() < end for start, end in consumed_spans):
            continue
        consumed_spans.append((m.start(), len(text)))
        break

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "BaseACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "BaseACPClient"):
        self.completions = _ACPChatCompletions(client)


def _coalesce_acp_args(
    acp_args: list[str] | None,
    args: list[str] | None,
    default_args_fn,
) -> list[str]:
    """Resolve ACP CLI args without treating ``[]`` as intentional.

    Call sites sometimes pass ``args=[]`` when runtime wiring forgot to
    forward the provider defaults (e.g. oneshot). An empty argv is never a
    valid ACP launch, so fall back to the provider's defaults instead of
    spawning a bare binary — and never leak a sibling provider's defaults
    via truthiness fallbacks.
    """
    if acp_args is not None and len(acp_args) > 0:
        return list(acp_args)
    if args is not None and len(args) > 0:
        return list(args)
    return list(default_args_fn())


class BaseACPClient:
    """Generic OpenAI-client-compatible facade for ACP subprocess backends."""

    # Subclasses override these so error messages and default model labels match
    # the actual backend.
    _acp_display_name = "ACP"
    _default_model_name = "acp"
    _install_hint = "Install the ACP CLI or set the command path."
    _acp_marker_base_url = "acp://"
    # This client wraps a long-lived subprocess and must not be closed on a
    # per-request path. run_agent.py honors this flag instead of provider-name
    # checks when deciding request-client lifecycle.
    shared_client = True
    # ACP can stream, but the complete-response path is cheaper for quiet/
    # subagent turns that have no display/TTS consumer.
    prefers_sync_without_consumers = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        provider: str | None = None,
        **_: Any,
    ):
        self._provider = provider or self._default_model_name
        self.api_key = api_key or self._default_model_name
        self.base_url = base_url or self._acp_marker_base_url
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or self._resolve_command()
        self._acp_args = _coalesce_acp_args(
            acp_args, args, lambda: self._resolve_args(self._acp_command)
        )
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._reuse_enabled = _acp_process_reuse_enabled()
        self._session_reuse_enabled = _acp_session_reuse_enabled()
        # Transport state for process reuse (guarded by _rpc_lock).
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._inbox: queue.Queue[dict[str, Any]] | None = None
        self._stderr_tail: deque[str] | None = None
        self._next_rpc_id = 0
        self._initialized = False
        self._spawn_count = 0  # test/metrics: how many times we Popen'd
        # Agent promptCapabilities from the last successful initialize.
        # Omitted flags are treated as unsupported per ACP.
        self._prompt_capabilities: dict[str, bool] = {
            "image": False,
            "audio": False,
            "embeddedContext": False,
        }
        # Session continuity state (same lock as transport).
        self._session_id: str | None = None
        self._session_history: list[dict[str, Any]] = []
        self._session_count = 0  # test/metrics: session/new calls
        self._session_continues = 0  # test/metrics: prompts reusing sessionId
        # Optional AIAgent (or compatible) for tool_progress / status / activity.
        # Bound after construction via bind_agent_activity() so create_openai_client
        # can wire Desktop/TUI progress without coupling constructors.
        self._activity_agent: Any = None
        # toolCallId -> last title (for completed events that omit title)
        self._tool_titles: dict[str, str] = {}

    def _resolve_command(self) -> str:
        """Return the ACP executable path. Subclasses may override."""
        try:
            from hermes_cli.auth import resolve_external_process_provider_credentials

            creds = resolve_external_process_provider_credentials(self._provider)
            command = str(creds.get("command") or "").strip()
            if command:
                return command
        except Exception:
            pass
        return self._provider

    def _resolve_args(self, command: str | None = None) -> list[str]:
        """Return the ACP argv. Subclasses may override."""
        try:
            from hermes_cli.auth import resolve_external_process_provider_credentials

            creds = resolve_external_process_provider_credentials(self._provider)
            cred_args = creds.get("args")
            if isinstance(cred_args, (list, tuple)) and cred_args:
                return list(cred_args)
        except Exception:
            pass
        return []

    def _is_deprecation_message(self, stderr_text: str) -> bool:
        """Return True if stderr indicates a deprecated CLI. Subclasses may override."""
        return False

    def bind_agent_activity(self, agent: Any) -> None:
        """Attach an AIAgent so ACP tool/session updates surface in the UI."""
        self._activity_agent = agent

    def _prompt_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Return tools to describe in the textual ACP prompt.

        ACP providers that expose tools through a native MCP connection can
        override this hook and leave the tool list out of the prompt. The
        original list is still passed to ``session/new`` for allowlisting.
        """
        return tools

    def _emit_acp_activity(
        self,
        event_type: str,
        name: str,
        preview: str,
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Push tool progress + status so the UI doesn't look frozen mid-turn."""
        agent = self._activity_agent
        if agent is None:
            return
        label = preview or name or "ACP activity"
        try:
            touch = getattr(agent, "_touch_activity", None)
            if callable(touch):
                touch(f"{self._acp_display_name}: {label}")
        except Exception:
            pass
        try:
            cb = getattr(agent, "tool_progress_callback", None)
            if callable(cb):
                cb(event_type, name, preview, args or {}, **kwargs)
        except Exception:
            pass
        # Lifecycle line for gateway/desktop status strip (spinner text).
        if event_type in {"tool.started", "tool.completed"}:
            try:
                emit = getattr(agent, "_emit_status", None)
                if callable(emit):
                    # ASCII-only markers — Windows CP* consoles choke on
                    # ellipsis/check glyphs in status paths.
                    verb = "... " if event_type == "tool.started" else "done "
                    emit(f"{self._acp_display_name} {verb}{label}")
            except Exception:
                pass

    def close(self) -> None:
        """Tear down the ACP subprocess and mark the client closed."""
        with self._rpc_lock:
            self._reset_transport(mark_closed=True)

    def interrupt(self) -> None:
        """Abort any in-flight ACP RPC by killing the warm subprocess.

        Does **not** wait on ``_rpc_lock`` so a blocked ``_rpc`` can observe
        ``poll() != None`` and raise. Leaves the client reusable
        (``is_closed`` stays False); the owning turn resets transport state.
        """
        with self._active_process_lock:
            proc = self._active_process
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except Exception:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _reset_session_state(self) -> None:
        self._session_id = None
        self._session_history = []

    def _reset_transport(self, *, mark_closed: bool = False) -> None:
        """Kill any live process and clear reuse state. Caller holds ``_rpc_lock``."""
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self._inbox = None
        self._stderr_tail = None
        self._next_rpc_id = 0
        self._initialized = False
        self._prompt_capabilities = {
            "image": False,
            "audio": False,
            "embeddedContext": False,
        }
        self._reset_session_state()
        if mark_closed:
            self.is_closed = True
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _process_alive(self) -> bool:
        proc = self._active_process
        return proc is not None and proc.poll() is None

    def _normalize_timeout(self, timeout: Any) -> float:
        if timeout is None:
            return _DEFAULT_TIMEOUT_SECONDS
        if isinstance(timeout, (int, float)):
            return float(timeout)
        # httpx.Timeout or similar — pick the largest component so the
        # subprocess has enough wall-clock time for the full response.
        _candidates = [
            getattr(timeout, attr, None)
            for attr in ("read", "write", "connect", "pool", "timeout")
        ]
        _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
        return max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        msg_list = [m for m in (messages or []) if isinstance(m, dict)]
        _effective_timeout = self._normalize_timeout(timeout)
        model_name = model or self._default_model_name

        if stream:
            return self._iter_stream_completion(
                msg_list,
                model=model_name,
                tools=tools,
                tool_choice=tool_choice,
                timeout_seconds=_effective_timeout,
            )

        response_text, reasoning_text = self._run_conversation_prompt(
            msg_list,
            model=model_name,
            tools=tools,
            tool_choice=tool_choice,
            timeout_seconds=_effective_timeout,
        )
        return self._build_completion(
            response_text,
            reasoning_text,
            model_name,
            tools=tools,
        )

    def _build_completion(
        self,
        response_text: str,
        reasoning_text: str,
        model_name: str,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> SimpleNamespace:
        tool_calls, cleaned_text = _extract_tool_calls_from_text(
            response_text,
            allowed_tool_names=_tool_names_from_schemas(tools),
        )
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model_name,
        )

    def _iter_stream_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        timeout_seconds: float,
    ):
        """Yield OpenAI-style stream chunks as ACP ``agent_message_chunk`` arrives."""
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        tool_call_filter = _ToolCallStreamFilter()

        def _on_text(chunk: str) -> None:
            visible = tool_call_filter.feed(chunk)
            if visible:
                events.put(("text", visible))

        def _on_reasoning(chunk: str) -> None:
            events.put(("reasoning", chunk))

        def _worker() -> None:
            try:
                text, reasoning = self._run_conversation_prompt(
                    messages,
                    model=model,
                    tools=tools,
                    tool_choice=tool_choice,
                    timeout_seconds=timeout_seconds,
                    on_text_chunk=_on_text,
                    on_reasoning_chunk=_on_reasoning,
                )
                events.put(("done", (text, reasoning)))
            except Exception as exc:
                events.put(("error", exc))

        worker = threading.Thread(target=_worker, daemon=True, name="acp-stream-worker")
        worker.start()

        role_sent = False
        while True:
            kind, payload = events.get()
            if kind == "text":
                delta = SimpleNamespace(
                    role="assistant" if not role_sent else None,
                    content=payload,
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                )
                role_sent = True
                yield SimpleNamespace(
                    choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
                    model=model,
                    usage=None,
                )
            elif kind == "reasoning":
                delta = SimpleNamespace(
                    role=None,
                    content=None,
                    tool_calls=None,
                    reasoning=payload,
                    reasoning_content=payload,
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
                    model=model,
                    usage=None,
                )
            elif kind == "error":
                worker.join(timeout=2)
                raise payload
            elif kind == "done":
                text, reasoning = payload
                tail = tool_call_filter.finish()
                if tail:
                    delta = SimpleNamespace(
                        role="assistant" if not role_sent else None,
                        content=tail,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    )
                    role_sent = True
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
                        model=model,
                        usage=None,
                    )
                completion = self._build_completion(
                    text,
                    reasoning or "",
                    model,
                    tools=tools,
                )
                # Emit terminal tool-call / finish frames after full parsing.
                for chunk in _completion_to_stream_chunks(completion):
                    choice0 = chunk.choices[0] if chunk.choices else None
                    delta = getattr(choice0, "delta", None) if choice0 else None
                    content = getattr(delta, "content", None) if delta else None
                    tool_calls = getattr(delta, "tool_calls", None) if delta else None
                    finish = getattr(choice0, "finish_reason", None) if choice0 else None
                    if content and not tool_calls and not finish and role_sent:
                        continue
                    if content and role_sent:
                        # Every visible byte was already emitted by the filter.
                        if delta is not None:
                            delta.content = None
                    yield chunk
                break

        worker.join(timeout=5)

    def _subprocess_env(self) -> dict[str, str]:
        """Environment for the ACP child process. Subclasses may extend."""
        return _build_subprocess_env()

    def _spawn_argv(self) -> list[str]:
        """Argv for the ACP child process. Subclasses may extend."""
        return [self._acp_command] + list(self._acp_args)

    def _prepare_for_model(self, model: str | None) -> None:
        """Hook before initialize/prompt so backends can rebind model state.

        Copilot ACP ignores Hermes model ids (no process-level model switch).
        DevinACPClient overrides this to set DEVIN_MODEL and respawn when needed.
        """
        del model

    def _apply_session_model(
        self,
        session_id: str,
        session: dict[str, Any],
        model: str | None,
        *,
        timeout_seconds: float,
    ) -> None:
        """Hook after ``session/new`` to bind the Hermes-selected model.

        Default no-op. DevinACPClient uses ``session/set_config_option``.
        """
        del session_id, session, model, timeout_seconds

    def _session_mcp_servers(
        self,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Return MCP servers to attach to a new ACP session.

        The base ACP adapter does not add any host-specific servers.  Providers
        that can use Hermes-owned MCP bridges may override this hook without
        changing the shared ACP lifecycle.
        """
        del tools
        return []

    def _spawn_process(self) -> subprocess.Popen[str]:
        label = self._acp_display_name
        try:
            # Force UTF-8 on the child pipes. On Windows the default console
            # encoding is often cp950/cp1252; Devin/Copilot ACP logs use UTF-8
            # (and occasionally raw bytes), so text=True alone raises
            # UnicodeDecodeError in the stderr reader and can leave the
            # transport half-dead while the UI shows no tokens.
            # Hide the console the CLI child would otherwise flash on Windows
            # (#56747). Hide-only — stdio pipes stay intact for the ACP wire.
            proc = subprocess.Popen(
                self._spawn_argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._acp_cwd,
                # Subclasses (Devin) override _subprocess_env for model binding.
                env=self._subprocess_env(),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start {label} command '{self._acp_command}'. "
                f"{self._install_hint}"
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError(f"{label} process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)
        self._inbox = inbox
        self._stderr_tail = stderr_tail
        self._next_rpc_id = 0
        self._initialized = False
        self._spawn_count += 1

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        threading.Thread(target=_stdout_reader, daemon=True).start()
        threading.Thread(target=_stderr_reader, daemon=True).start()
        return proc

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
    ) -> Any:
        """Send one JSON-RPC request on the live transport. Caller holds ``_rpc_lock``."""
        label = self._acp_display_name
        proc = self._active_process
        inbox = self._inbox
        stderr_tail = self._stderr_tail
        if proc is None or inbox is None or stderr_tail is None or proc.stdin is None:
            raise RuntimeError(f"{label} transport is not ready.")

        self._next_rpc_id += 1
        request_id = self._next_rpc_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                msg = inbox.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._handle_server_message(
                msg,
                process=proc,
                cwd=self._acp_cwd,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                on_text_chunk=on_text_chunk,
                on_reasoning_chunk=on_reasoning_chunk,
            ):
                continue

            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                raise RuntimeError(
                    f"{label} {method} failed: {_acp_rpc_error_message(err)}"
                )
            return msg.get("result")

        stderr_text = "\n".join(stderr_tail).strip()
        if proc.poll() is not None and stderr_text:
            if self._is_deprecation_message(stderr_text):
                raise RuntimeError(
                    "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                    "(github.com/github/copilot-cli), but the binary it just "
                    "spawned is the deprecated `gh copilot` extension.\n\n"
                    "Install the new CLI:\n"
                    "  npm install -g @github/copilot\n"
                    "  # then verify with: copilot --help\n\n"
                    "If `copilot` already resolves to the new CLI but you still see this,\n"
                    "point Hermes at it explicitly:\n"
                    "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
                    "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\n"
                    "directly with a Copilot subscription token) via `hermes setup`.\n\n"
                    f"Original error:\n{stderr_text}"
                )
            raise RuntimeError(f"{label} process exited early: {stderr_text}")
        raise TimeoutError(f"Timed out waiting for {label} response to {method}.")

    def _record_initialize_result(self, init: dict[str, Any] | None) -> None:
        """Cache agent capabilities advertised by ACP ``initialize``."""
        self._prompt_capabilities = _parse_prompt_capabilities(init)

    def _ensure_initialized(self, *, timeout_seconds: float) -> None:
        """Spawn (if needed) and run ACP ``initialize`` once per process."""
        if self._process_alive() and self._initialized:
            return
        if not self._process_alive():
            self._reset_transport(mark_closed=False)
            self._spawn_process()
        init = self._rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {
                        "readTextFile": True,
                        "writeTextFile": True,
                    }
                },
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            },
            timeout_seconds=timeout_seconds,
        ) or {}
        self._record_initialize_result(init)
        self._initialized = True

    def _run_conversation_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        timeout_seconds: float,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
    ) -> tuple[str, str]:
        """Run one completion, reusing process and optionally ACP session."""
        label = self._acp_display_name
        with self._rpc_lock:
            try:
                # Allow backends (Devin) to rebind process-level model before
                # initialize — may tear down a warm process when the model changes.
                self._prepare_for_model(model)
                self._ensure_initialized(timeout_seconds=timeout_seconds)

                prefix_len = 0
                continue_session = False
                if (
                    self._session_reuse_enabled
                    and self._session_id
                    and self._process_alive()
                    and self._initialized
                ):
                    prefix_len = _common_message_prefix_len(self._session_history, messages)
                    if 0 < prefix_len < len(messages):
                        continue_session = True

                if continue_session:
                    delta_messages = messages[prefix_len:]
                    prompt_text = _format_messages_as_prompt(
                        delta_messages,
                        model=model,
                        tools=self._prompt_tools(tools),
                        tool_choice=tool_choice,
                        continuation=True,
                    )
                    media_blocks = _extract_acp_media_blocks(delta_messages)
                    session_id = self._session_id
                    assert session_id is not None
                    try:
                        text, reasoning = self._session_prompt(
                            session_id,
                            prompt_text,
                            timeout_seconds=timeout_seconds,
                            on_text_chunk=on_text_chunk,
                            on_reasoning_chunk=on_reasoning_chunk,
                            media_blocks=media_blocks,
                        )
                        self._session_continues += 1
                        self._session_history = list(messages)
                        return text, reasoning
                    except Exception:
                        # Session may have expired — fall through to a fresh
                        # session/new with the full transcript on the same process.
                        self._reset_session_state()

                prompt_text = _format_messages_as_prompt(
                    messages,
                    model=model,
                    tools=self._prompt_tools(tools),
                    tool_choice=tool_choice,
                    continuation=False,
                )
                media_blocks = _extract_acp_media_blocks(messages)
                session = self._rpc(
                    "session/new",
                    {
                        "cwd": self._acp_cwd,
                        "mcpServers": self._session_mcp_servers(tools),
                    },
                    timeout_seconds=timeout_seconds,
                ) or {}
                session_id = str(session.get("sessionId") or "").strip()
                if not session_id:
                    raise RuntimeError(f"{label} did not return a sessionId.")
                self._session_id = session_id
                self._session_count += 1
                # Subclasses (Devin) bind model via ACP session/set_config_option;
                # CLI --model is ignored by some ACP agents.
                self._apply_session_model(
                    session_id,
                    session,
                    model,
                    timeout_seconds=timeout_seconds,
                )

                text, reasoning = self._session_prompt(
                    session_id,
                    prompt_text,
                    timeout_seconds=timeout_seconds,
                    on_text_chunk=on_text_chunk,
                    on_reasoning_chunk=on_reasoning_chunk,
                    media_blocks=media_blocks,
                )
                self._session_history = list(messages)
                return text, reasoning
            except Exception:
                # Drop a possibly-poisoned transport so the next call gets a
                # clean process rather than fighting a half-dead inbox.
                self._reset_transport(mark_closed=False)
                raise
            finally:
                if not self._reuse_enabled:
                    self._reset_transport(mark_closed=True)

    def _session_prompt(
        self,
        session_id: str,
        prompt_text: str,
        *,
        timeout_seconds: float,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
        media_blocks: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Immediate status so the UI shows activity before the first token/tool.
        # Devin often spends 10-60s connecting user MCP servers after
        # session/prompt with zero agent_message_chunk — without this the
        # chat looks frozen / "no opening reply".
        try:
            agent = self._activity_agent
            emit = getattr(agent, "_emit_status", None) if agent else None
            touch = getattr(agent, "_touch_activity", None) if agent else None
            msg = f"{self._acp_display_name} working (may connect MCP tools first)..."
            if callable(touch):
                touch(msg)
            if callable(emit):
                emit(msg)
        except Exception:
            pass
        session_deadline = time.monotonic() + timeout_seconds
        prompt_blocks = _build_acp_prompt_blocks(
            prompt_text,
            media_blocks,
            prompt_capabilities=self._prompt_capabilities,
        )
        self._rpc(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": prompt_blocks,
            },
            timeout_seconds=timeout_seconds,
            text_parts=text_parts,
            reasoning_parts=reasoning_parts,
            on_text_chunk=on_text_chunk,
            on_reasoning_chunk=on_reasoning_chunk,
        )
        # session/prompt may return stopReason before the final
        # session/update agent_message_chunk is flushed; drain trailing
        # updates until the response text is stable.
        self._drain_session_prompt_chunks(
            text_parts,
            reasoning_parts,
            on_text_chunk=on_text_chunk,
            on_reasoning_chunk=on_reasoning_chunk,
            timeout_seconds=max(0.0, session_deadline - time.monotonic()),
        )
        return "".join(text_parts), "".join(reasoning_parts)

    def _drain_session_prompt_chunks(
        self,
        text_parts: list[str],
        reasoning_parts: list[str],
        *,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
        timeout_seconds: float,
    ) -> None:
        """Drain trailing session/update chunks after session/prompt returns.

        ACP servers (Grok, Devin, Copilot) may send the JSON-RPC response for
        session/prompt before the final agent_message_chunk stream is flushed.
        Wait until no new chunk arrives for a short stable window.
        """
        proc = self._active_process
        inbox = self._inbox
        if proc is None or inbox is None:
            return

        # If the prompt response won the race with the first notification,
        # waiting for only the normal quiet window can return an empty answer
        # while the real answer is still in flight.  In that case wait for the
        # first chunk for a bounded grace period, then switch to the short
        # quiet window once output has started.  This prevents a late
        # first-turn chunk from being consumed by the next prompt.
        content_seen = bool(text_parts or reasoning_parts)
        idle_seconds = _ACP_POST_RESPONSE_IDLE_SECONDS
        quiet_window = idle_seconds * _ACP_POST_RESPONSE_STABLE_CHECKS
        overall_deadline = time.monotonic() + timeout_seconds
        first_chunk_deadline = (
            min(overall_deadline, time.monotonic() + _ACP_FIRST_CHUNK_GRACE_SECONDS)
            if not content_seen
            else None
        )
        quiet_deadline = min(overall_deadline, time.monotonic() + quiet_window)
        stable_count = 0
        while True:
            now = time.monotonic()
            if not content_seen:
                if first_chunk_deadline is None or now >= first_chunk_deadline:
                    break
                deadline = first_chunk_deadline
            else:
                if now >= quiet_deadline:
                    break
                deadline = quiet_deadline

            remaining = min(idle_seconds, deadline - now)
            if remaining <= 0:
                break
            try:
                msg = inbox.get(timeout=remaining)
            except queue.Empty:
                if not content_seen:
                    # No output yet: the ACP server may have acknowledged the
                    # prompt before publishing its first agent chunk.  Keep
                    # listening until the first-chunk grace period expires.
                    continue
                stable_count += 1
                if stable_count >= _ACP_POST_RESPONSE_STABLE_CHECKS:
                    break
                continue

            stable_count = 0
            before_content = len(text_parts) + len(reasoning_parts)
            if self._handle_server_message(
                msg,
                process=proc,
                cwd=self._acp_cwd,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                on_text_chunk=on_text_chunk,
                on_reasoning_chunk=on_reasoning_chunk,
            ):
                after_content = len(text_parts) + len(reasoning_parts)
                if after_content > before_content:
                    content_seen = True
                    quiet_deadline = min(overall_deadline, time.monotonic() + quiet_window)
                continue

            # Not a notification we can drain; put it back for the next RPC.
            try:
                inbox.put(msg)
            except Exception:
                pass
            break

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        """Low-level single-blob prompt (tests / callers that skip message lists)."""
        return self._run_conversation_prompt(
            [{"role": "user", "content": prompt_text}],
            model=None,
            tools=None,
            tool_choice=None,
            timeout_seconds=timeout_seconds,
        )

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            if not isinstance(update, dict):
                update = {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text:
                if text_parts is not None:
                    text_parts.append(chunk_text)
                if on_text_chunk is not None:
                    try:
                        on_text_chunk(chunk_text)
                    except Exception:
                        pass
            elif kind == "agent_thought_chunk" and chunk_text:
                if reasoning_parts is not None:
                    reasoning_parts.append(chunk_text)
                if on_reasoning_chunk is not None:
                    try:
                        on_reasoning_chunk(chunk_text)
                    except Exception:
                        pass
            elif kind in {"tool_call", "tool_call_update"}:
                self._handle_tool_session_update(kind, update)
            elif kind in {
                "available_commands_update",
                "current_mode_update",
                "config_option_update",
                "plan",
            }:
                # Heartbeat so long turns still touch activity even without tools.
                try:
                    agent = self._activity_agent
                    touch = getattr(agent, "_touch_activity", None) if agent else None
                    if callable(touch):
                        touch(f"{self._acp_display_name}: {kind.replace('_', ' ')}")
                except Exception:
                    pass
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            if _acp_auto_approve_enabled():
                response = _permission_auto_selected(
                    message_id, params.get("options")
                )
                # Surface the pending tool in the activity strip.
                tool_call = params.get("toolCall") or params.get("tool_call") or {}
                if isinstance(tool_call, dict):
                    preview = _tool_update_text_preview(tool_call)
                    self._emit_acp_activity(
                        "tool.started",
                        _tool_kind_name(tool_call),
                        preview or "awaiting permission (auto-approved)",
                        {"toolCallId": tool_call.get("toolCallId")},
                    )
            else:
                response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                denied = get_write_denied_error(str(path))
                if denied:
                    raise PermissionError(denied)
                # Approval-gated paths (e.g. ~/.ssh/config) are not hard-denied
                # for interactive tools, but the ACP shim has no human channel
                # to confirm the write — fail closed here.
                if is_write_approval_required(str(path)):
                    raise PermissionError(
                        f"Write denied: '{path}' requires interactive approval "
                        "and cannot be written through the ACP file bridge."
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""), encoding="utf-8")
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True

    def _handle_tool_session_update(self, kind: str, update: dict[str, Any]) -> None:
        """Map ACP tool_call / tool_call_update notifications onto Hermes progress."""
        tool_id = str(update.get("toolCallId") or update.get("tool_call_id") or "").strip()
        title = str(update.get("title") or "").strip()
        if title and tool_id:
            self._tool_titles[tool_id] = title
        elif tool_id and not title:
            title = self._tool_titles.get(tool_id, "")

        status = str(update.get("status") or "").strip().lower()
        preview = _tool_update_text_preview(
            {**update, "title": title or update.get("title")}
        )
        name = _tool_kind_name(update)
        args: dict[str, Any] = {
            "toolCallId": tool_id or None,
            "kind": update.get("kind"),
            "status": status or None,
        }
        locations = update.get("locations")
        if isinstance(locations, list) and locations:
            paths = []
            for loc in locations[:3]:
                if isinstance(loc, dict) and loc.get("path"):
                    paths.append(str(loc["path"]))
            if paths:
                args["paths"] = paths

        if kind == "tool_call" or status in {"", "pending", "in_progress"}:
            event = "tool.started"
            if status == "in_progress" and kind == "tool_call_update":
                # Keep as started so UIs that only listen for started still refresh
                # the preview; duration is unknown mid-flight.
                event = "tool.started"
            if status in {"completed", "failed"}:
                event = "tool.completed"
            self._emit_acp_activity(
                event,
                name,
                preview,
                args,
                is_error=(status == "failed"),
            )
            return

        if status in {"completed", "failed"}:
            self._emit_acp_activity(
                "tool.completed",
                name,
                preview,
                args,
                is_error=(status == "failed"),
            )
            if tool_id:
                self._tool_titles.pop(tool_id, None)
            return

        # Unknown status — still heartbeat so the UI doesn't freeze.
        self._emit_acp_activity("tool.started", name, preview, args)
