import os
import unittest
from unittest import mock

import config
from tests.fakes import FakeClient, make_response


class TestConfig(unittest.TestCase):

    def test_api_key_from_environment(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "  abc123  "}, clear=True):
            self.assertEqual(config.get_api_key(), "abc123")

    def test_google_api_key_fallback(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "g-key"}, clear=True):
            self.assertEqual(config.get_api_key(), "g-key")

    def test_streamlit_secret_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(config, "_read_streamlit_secret", return_value="s-key"):
            self.assertEqual(config.get_api_key(), "s-key")

    def test_missing_key_raises_config_error(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(config, "_read_streamlit_secret", return_value=None):
            with self.assertRaises(config.ConfigError):
                config.create_client()

    def test_model_overrides(self):
        env = {"GEMINI_MODEL": "gemini-x", "GEMINI_JUDGE_MODEL": "gemini-y"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(config, "_read_streamlit_secret", return_value=None):
            settings = config.get_settings()
        self.assertEqual(settings.model, "gemini-x")
        self.assertEqual(settings.judge_model, "gemini-y")
        self.assertIsNone(settings.api_key)

    def test_check_connection_uses_client(self):
        client = FakeClient([make_response(" ready \n")])
        self.assertEqual(config.check_connection(client=client, model="m"), "ready")


if __name__ == "__main__":
    unittest.main()
