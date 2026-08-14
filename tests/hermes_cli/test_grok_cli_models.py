"""Grok CLI model auto-discovery (`grok models`)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import providers
from hermes_cli.models import parse_grok_cli_available_models, provider_model_ids


SAMPLE_STDOUT = (
    "Available models:\n"
    "  * grok-4.5 (default)\n"
    "  - grok-composer-2.5-fast\n"
    "  - grok-build-0.1\n"
)


class TestParseGrokAvailable(unittest.TestCase):
    def test_parses_available_models(self):
        models = parse_grok_cli_available_models(SAMPLE_STDOUT)
        self.assertEqual(
            models,
            ["grok-4.5", "grok-composer-2.5-fast", "grok-build-0.1"],
        )

    def test_empty_when_missing(self):
        self.assertEqual(parse_grok_cli_available_models("no available here"), [])
        self.assertEqual(parse_grok_cli_available_models(""), [])

    def test_dedupes_preserving_order(self):
        text = "Available models:\n  * grok-4.5\n  - grok-4.5\n  - grok-4.3\n"
        self.assertEqual(parse_grok_cli_available_models(text), ["grok-4.5", "grok-4.3"])


class TestFetchGrokCliModels(unittest.TestCase):
    def setUp(self):
        self.grok = providers.get_provider_profile("grok-acp")
        self.assertIsNotNone(self.grok)

    def test_fetch_parses_output(self):
        mock_proc = MagicMock()
        mock_proc.stderr = ""
        mock_proc.stdout = SAMPLE_STDOUT
        with patch("subprocess.run", return_value=mock_proc) as run:
            with patch("shutil.which", return_value="/bin/grok"):
                models = self.grok.fetch_models(timeout=1.0)
        self.assertEqual(
            models,
            ["grok-4.5", "grok-composer-2.5-fast", "grok-build-0.1"],
        )
        argv = run.call_args[0][0]
        self.assertEqual(argv[0], "/bin/grok")
        self.assertEqual(argv[1], "models")

    def test_fetch_returns_empty_on_timeout(self):
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="grok", timeout=1),
        ):
            with patch("shutil.which", return_value="/bin/grok"):
                self.assertEqual(self.grok.fetch_models(timeout=1.0), [])

    def test_provider_model_ids_uses_live(self):
        with patch.object(
            self.grok, "fetch_models", return_value=["grok-4.5", "grok-composer-2.5-fast"]
        ):
            ids = provider_model_ids("grok-acp", force_refresh=True)
        self.assertEqual(ids, ["grok-4.5", "grok-composer-2.5-fast"])

    def test_provider_model_ids_falls_back_to_curated(self):
        with patch.object(self.grok, "fetch_models", return_value=[]):
            ids = provider_model_ids("grok-acp", force_refresh=True)
        self.assertIn("grok-4.5", ids)
        self.assertIn("grok-build-0.1", ids)
        self.assertIn("grok-acp", ids)


if __name__ == "__main__":
    unittest.main()
