"""Tests for ACP client base helpers shared across ACP providers."""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent.copilot_acp_client import CopilotACPClient


# 1x1 transparent PNG — smallest valid image payload for content-part tests.
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
