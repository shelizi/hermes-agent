"""Focused regressions for the Copilot ACP shim safety layer."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent.copilot_acp_client import CopilotACPClient


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class CopilotACPClientSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = CopilotACPClient(acp_cwd="/tmp")



    @unittest.skip("_create_chat_completion stream path uses _run_conversation_prompt in BaseACPClient; test needs refactor")
    def test_stream_true_preserves_tool_call_deltas(self) -> None:
        tool_response = (
            "<tool_call>"
            '{"id":"call_read","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            "</tool_call>"
        )

        with patch.object(self.client, "_run_conversation_prompt", return_value=(tool_response, "")):
            stream = self.client._create_chat_completion(
                model="copilot-acp",
                messages=[{"role": "user", "content": "read README.md"}],
                stream=True,
            )

        chunks = list(stream)
        delta = chunks[0].choices[0].delta
        self.assertIsNone(delta.content)
        self.assertEqual(chunks[0].choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(delta.tool_calls), 1)
        tool_delta = delta.tool_calls[0]
        self.assertEqual(tool_delta.index, 0)
        self.assertEqual(tool_delta.id, "call_read")
        self.assertEqual(tool_delta.function.name, "read_file")
        self.assertEqual(
            json.loads(tool_delta.function.arguments),
            {"path": "README.md"},
        )
        self.assertEqual(chunks[1].choices, [])


    def _dispatch(self, message: dict, *, cwd: str) -> dict:
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            message,
            process=process,
            cwd=cwd,
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        payload = process.stdin.getvalue().strip()
        self.assertTrue(payload)
        return json.loads(payload)



    def test_read_text_file_redacts_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret_file = root / "config.env"
            secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")

            # agent.redact snapshots HERMES_REDACT_SECRETS at import time into
            # _REDACT_ENABLED, so patching os.environ is a no-op. Flip the
            # module-level constant directly for the duration of the call.
            with patch("agent.redact._REDACT_ENABLED", True):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "fs/read_text_file",
                        "params": {"path": str(secret_file)},
                    },
                    cwd=str(root),
                )

        content = ((response.get("result") or {}).get("content") or "")
        self.assertNotIn("abc123def456", content)
        self.assertIn("OPENAI_API_KEY=", content)

    def test_fs_read_text_file_decodes_as_utf8_under_non_utf8_locale(self) -> None:
        """Regression for #18637 (bug 2): fs/read_text_file used
        ``path.read_text()`` with no explicit encoding, so on Windows
        GBK/CP932/CP949 locales the Copilot read_file tool crashed on any
        source file with non-ASCII content (e.g. a CJK comment, an em dash,
        or UTF-8 BOM)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "note.md"
            target.write_text("# 中文标题\nem dash — here\n", encoding="utf-8")

            original_read_text = Path.read_text

            def strict_read_text(self, encoding=None, errors=None, **kwargs):
                if self == target and encoding != "utf-8":
                    raise UnicodeDecodeError(
                        "gbk", b"\x94", 0, 1, "illegal multibyte sequence"
                    )
                return original_read_text(
                    self, encoding=encoding, errors=errors, **kwargs
                )

            with patch.object(Path, "read_text", strict_read_text):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 10,
                        "method": "fs/read_text_file",
                        "params": {"path": str(target)},
                    },
                    cwd=str(root),
                )

        self.assertNotIn("error", response)
        content = ((response.get("result") or {}).get("content") or "")
        self.assertIn("中文标题", content)
        self.assertIn("em dash —", content)



    def test_write_text_file_respects_safe_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"

            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(outside),
                            "content": "should-not-write",
                        },
                    },
                    cwd=str(root),
                )

        self.assertIn("error", response)
        self.assertIn("HERMES_WRITE_SAFE_ROOT", str(response["error"]))
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()


# ── HOME env propagation tests (from PR #11285) ─────────────────────

from unittest.mock import patch as _patch
import pytest


def _make_home_client(tmp_path):
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )


def _fake_popen_capture(captured):
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise FileNotFoundError("copilot not found")
    return _fake


def test_run_prompt_preserves_real_home_when_profile_home_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()

    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Hermeticity: an ambient HERMES_REAL_HOME (exported by Hermes' own
    # terminal contract on dev boxes) outranks HOME in the candidate ladder,
    # and an ambient TERMINAL_HOME_MODE would change the policy under test.
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
    # Hermeticity: get_subprocess_home()'s auto mode prefers the profile home
    # when is_container() is True — on a containerized CI runner that real
    # probe flips the resolution this test asserts. The host/VM branch is the
    # contract under test; pin containment off.
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)

    captured = {}
    client = _make_home_client(tmp_path)

    # Hermeticity: the --acp support probe (PR #87308) calls subprocess.run
    # before Popen; stub it inconclusive so no real CLI on the host box can
    # flip the resolution this test asserts.
    with _patch("agent.copilot_acp_client.subprocess.run", side_effect=FileNotFoundError):
        with _patch("agent.acp_client_base.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
            with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
                client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(real_home)
    assert captured["kwargs"]["env"]["HERMES_REAL_HOME"] == str(real_home)


def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    # Hermeticity: the --acp support probe (PR #87308) calls subprocess.run
    # before Popen; stub it inconclusive so no real CLI on the host box can
    # flip the resolution this test asserts.
    with _patch("agent.copilot_acp_client.subprocess.run", side_effect=FileNotFoundError):
        with _patch("agent.acp_client_base.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
            with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
                client._run_prompt("hello", timeout_seconds=1)

    assert "env" in captured["kwargs"]
    assert captured["kwargs"]["env"]["HOME"]


# ── --acp support probe tests (PR #87308 / issue #87309) ────────────

import subprocess as _subprocess

from agent.copilot_acp_client import _ACP_PROBE_CACHE, _acp_supported


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    _ACP_PROBE_CACHE.clear()
    yield
    _ACP_PROBE_CACHE.clear()


def _completed(returncode=0, stdout=""):
    return _subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_probe_true_when_help_advertises_acp():
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        return_value=_completed(stdout="Usage: copilot [--acp] [--stdio]"),
    ):
        assert _acp_supported("copilot", ["--acp", "--stdio"]) is True


def test_probe_false_when_help_lacks_acp_and_run_prompt_fast_fails(tmp_path):
    client = _make_home_client(tmp_path)
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        return_value=_completed(stdout="Usage: claude [--print] [--model]"),
    ):
        with pytest.raises(RuntimeError, match="ACP transport not supported"):
            client._run_prompt("hello", timeout_seconds=1)


def test_probe_inconclusive_falls_through_to_spawn_error(tmp_path):
    """Missing binary: probe must NOT mask the established spawn error."""
    client = _make_home_client(tmp_path)
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        side_effect=FileNotFoundError("copilot not found"),
    ):
        with _patch(
            "agent.acp_client_base.subprocess.Popen",
            side_effect=FileNotFoundError("copilot not found"),
        ):
            with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
                client._run_prompt("hello", timeout_seconds=1)


def test_probe_result_cached_per_binary_path():
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        return_value=_completed(stdout="Usage: copilot [--acp]"),
    ) as run_mock:
        assert _acp_supported("copilot", ["--acp"]) is True
        assert _acp_supported("copilot", ["--acp"]) is True
    assert run_mock.call_count == 1


def test_probe_inconclusive_not_cached():
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        side_effect=FileNotFoundError,
    ) as run_mock:
        assert _acp_supported("copilot", ["--acp"]) is None
        assert _acp_supported("copilot", ["--acp"]) is None
    assert run_mock.call_count == 2  # inconclusive verdicts retry


def test_probe_skipped_for_custom_args_without_acp():
    with _patch("agent.copilot_acp_client.subprocess.run") as run_mock:
        assert _acp_supported("mycli", ["--custom-transport"]) is True
    run_mock.assert_not_called()
_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)
_ONE_PX_B64 = base64.b64encode(_ONE_PX_PNG).decode("ascii")


class TestAcpContentParts(unittest.TestCase):
    """OpenAI multimodal parts → ACP ContentBlocks for session/prompt."""

    def test_data_url_image_part_becomes_acp_image_block(self) -> None:
        from agent.acp_client_base import _openai_image_part_to_acp_block

        block = _openai_image_part_to_acp_block(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_ONE_PX_B64}",
                },
            }
        )
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["mimeType"], "image/png")
        self.assertEqual(block["data"], _ONE_PX_B64)

    def test_local_file_image_part_is_inlined(self) -> None:
        from agent.acp_client_base import _openai_image_part_to_acp_block

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shot.png"
            path.write_bytes(_ONE_PX_PNG)
            block = _openai_image_part_to_acp_block(
                {
                    "type": "image_url",
                    "image_url": {"url": str(path)},
                }
            )
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["mimeType"], "image/png")
        self.assertEqual(block["data"], _ONE_PX_B64)
        self.assertTrue(str(block.get("uri") or "").startswith("file:"))

    def test_extract_media_from_messages_collects_images_and_files(self) -> None:
        from agent.acp_client_base import _extract_acp_media_blocks

        with tempfile.TemporaryDirectory() as tmpdir:
            notes = Path(tmpdir) / "notes.md"
            notes.write_text("# hi\n", encoding="utf-8")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_ONE_PX_B64}",
                            },
                        },
                        {
                            "type": "file",
                            "name": "notes.md",
                            "path": str(notes),
                            "mimeType": "text/markdown",
                        },
                    ],
                }
            ]
            blocks = _extract_acp_media_blocks(messages)

        types = [b["type"] for b in blocks]
        self.assertIn("image", types)
        self.assertIn("resource_link", types)
        image = next(b for b in blocks if b["type"] == "image")
        self.assertEqual(image["data"], _ONE_PX_B64)
        link = next(b for b in blocks if b["type"] == "resource_link")
        self.assertEqual(link["name"], "notes.md")
        self.assertTrue(str(link["uri"]).startswith("file:"))

    def test_build_prompt_blocks_gates_image_on_capability(self) -> None:
        from agent.acp_client_base import _build_acp_prompt_blocks

        media = [
            {
                "type": "image",
                "mimeType": "image/png",
                "data": _ONE_PX_B64,
            },
            {
                "type": "resource_link",
                "uri": "file:///tmp/a.md",
                "name": "a.md",
            },
        ]
        with_cap = _build_acp_prompt_blocks(
            "hello",
            media,
            prompt_capabilities={"image": True},
        )
        self.assertEqual(len(with_cap), 3)
        self.assertEqual(with_cap[0]["type"], "text")
        self.assertEqual(with_cap[1]["type"], "image")
        self.assertEqual(with_cap[2]["type"], "resource_link")

        without_cap = _build_acp_prompt_blocks(
            "hello",
            media,
            prompt_capabilities={"image": False},
        )
        # text (+ capability note) + resource_link only
        self.assertEqual(len(without_cap), 2)
        self.assertEqual(without_cap[0]["type"], "text")
        self.assertIn("promptCapabilities", without_cap[0]["text"])
        self.assertEqual(without_cap[1]["type"], "resource_link")

    def test_render_message_content_keeps_image_placeholder(self) -> None:
        from agent.acp_client_base import _render_message_content

        rendered = _render_message_content(
            [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_ONE_PX_B64}",
                    },
                },
            ]
        )
        self.assertIn("What is this?", rendered)
        self.assertIn("[Image attached", rendered)
        # Must not dump the full base64 into the text transcript.
        self.assertNotIn(_ONE_PX_B64, rendered)

    def test_parse_prompt_capabilities_from_initialize(self) -> None:
        from agent.acp_client_base import _parse_prompt_capabilities

        caps = _parse_prompt_capabilities(
            {
                "agentCapabilities": {
                    "promptCapabilities": {
                        "image": True,
                        "embeddedContext": True,
                    }
                }
            }
        )
        self.assertTrue(caps["image"])
        self.assertFalse(caps["audio"])
        self.assertTrue(caps["embeddedContext"])

        empty = _parse_prompt_capabilities({})
        self.assertFalse(empty["image"])

    def test_session_prompt_includes_image_blocks_when_capable(self) -> None:
        client = CopilotACPClient(acp_cwd="/tmp")
        client._prompt_capabilities = {
            "image": True,
            "audio": False,
            "embeddedContext": False,
        }
        captured: dict[str, Any] = {}

        def fake_rpc(method, params, **kwargs):
            captured["method"] = method
            captured["params"] = params
            return {"stopReason": "end_turn"}

        with patch.object(client, "_rpc", side_effect=fake_rpc), patch.object(
            client, "_drain_session_prompt_chunks"
        ):
            client._session_prompt(
                "sess-1",
                "describe this",
                timeout_seconds=5,
                media_blocks=[
                    {
                        "type": "image",
                        "mimeType": "image/png",
                        "data": _ONE_PX_B64,
                    }
                ],
            )

        self.assertEqual(captured["method"], "session/prompt")
        prompt = captured["params"]["prompt"]
        self.assertEqual(prompt[0]["type"], "text")
        self.assertEqual(prompt[0]["text"], "describe this")
        self.assertEqual(prompt[1]["type"], "image")
        self.assertEqual(prompt[1]["data"], _ONE_PX_B64)
