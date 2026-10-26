"""End-to-end test of the Streamlit app with a fake Gemini client."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

from tests.fakes import EchoClient


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestAppEndToEnd(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        env = {"GEMINI_API_KEY": "test-key", "PROMPTBENCH_DATA_DIR": self._tmp.name}
        self._env = mock.patch.dict(os.environ, env)
        self._env.start()
        self.client = EchoClient()
        self._client = mock.patch("config.create_client", return_value=self.client)
        self._client.start()

    def tearDown(self):
        self._client.stop()
        self._env.stop()
        self._tmp.cleanup()

    def _button(self, app, label_start):
        return next(b for b in app.button if b.label.startswith(label_start))

    def test_generate_run_and_view_history(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertFalse(app.exception)

        app.text_area[0].input("Explain machine learning to a beginner.")
        app.multiselect[0].set_value(["zero_shot", "structured_output"])
        self._button(app, "Generate prompt variants").click().run()
        self.assertFalse(app.exception)

        self._button(app, "Run 2 prompts").click().run()
        self.assertFalse(app.exception)

        # Two generation calls plus two judge calls.
        self.assertEqual(len(self.client.models.calls), 4)

        headers = [h.value for h in app.header]
        self.assertIn("4. Experiment results", headers)
        self.assertIn("Saved results", headers)

        table = app.dataframe[0].value
        self.assertEqual(list(table["technique"]), ["zero_shot", "structured_output"])
        self.assertEqual(list(table["evaluation_score"]), [8.0, 8.0])
        self.assertEqual(list(table["input_tokens"]), [15, 15])

        saved = list(Path(self._tmp.name).glob("*.json"))
        self.assertEqual(len(saved), 1)

    def test_run_button_disabled_without_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
            app = AppTest.from_file(APP_PATH, default_timeout=30).run()
            app.text_area[0].input("Explain ML.")
            self._button(app, "Generate prompt variants").click().run()

            self.assertTrue(self._button(app, "Run ").disabled)
            self.assertTrue(any("No Gemini API key" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
