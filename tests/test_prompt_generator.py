
import unittest

from prompt_generator import generate_prompts, TECHNIQUES


class TestPromptGenerator(unittest.TestCase):

    def test_generates_all_techniques(self):
        result = generate_prompts("Explain machine learning.")

        self.assertEqual(set(result), set(TECHNIQUES))
        self.assertEqual(len(result), 6)

    def test_generates_selected_techniques(self):
        result = generate_prompts(
            "Explain machine learning.",
            ["zero_shot", "role_prompting"],
        )

        self.assertEqual(
            set(result),
            {"zero_shot", "role_prompting"},
        )

    def test_zero_shot_preserves_original_task(self):
        task = "Explain machine learning."
        result = generate_prompts(task, ["zero_shot"])

        self.assertEqual(result["zero_shot"], task)

    def test_rejects_empty_task(self):
        with self.assertRaises(ValueError):
            generate_prompts("   ")

    def test_rejects_unknown_technique(self):
        with self.assertRaises(ValueError):
            generate_prompts("Explain AI.", ["unknown"])

    def test_rejects_empty_selection(self):
        with self.assertRaises(ValueError):
            generate_prompts("Explain AI.", [])


if __name__ == "__main__":
    unittest.main()
