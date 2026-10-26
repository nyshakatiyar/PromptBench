import json
import unittest

from validators import (
    check_constraints,
    check_reference,
    check_structured_output,
    token_f1,
    validate_response,
)


VALID_STRUCTURED = json.dumps(
    {"answer": "Positive", "key_points": ["praise"], "limitations": []}
)


class TestStructuredOutput(unittest.TestCase):

    def test_valid_structured_output(self):
        self.assertEqual(check_structured_output(VALID_STRUCTURED), [])

    def test_invalid_json(self):
        issues = check_structured_output('{"answer": "x",')
        self.assertTrue(issues[0].startswith("Invalid JSON"))

    def test_missing_and_wrong_type_keys(self):
        issues = check_structured_output('{"answer": "x", "key_points": "one"}')
        self.assertIn('JSON key "key_points" should be a list.', issues)
        self.assertIn('Missing JSON key "limitations".', issues)

    def test_code_fences_are_flagged(self):
        issues = check_structured_output(f"```json\n{VALID_STRUCTURED}\n```")
        self.assertEqual(len(issues), 1)
        self.assertIn("code fences", issues[0])


class TestConstraints(unittest.TestCase):

    def test_word_limit(self):
        self.assertEqual(check_constraints("Explain in 5 words", "One two three four five"), [])
        issues = check_constraints("Explain in 5 words", "One two three four five six")
        self.assertIn("limit is 5", issues[0])

    def test_word_limit_in_words(self):
        issues = check_constraints("Answer in no more than three words", "a b c d")
        self.assertEqual(len(issues), 1)

    def test_sentence_limit(self):
        issues = check_constraints("Summarize in 2 sentences.", "One. Two. Three.")
        self.assertIn("3 sentences", issues[0])

    def test_no_limits_passes(self):
        self.assertEqual(check_constraints("Explain ML", "Long answer " * 50), [])

    def test_empty_response(self):
        self.assertEqual(check_constraints("anything", "  "), ["Response is empty."])


class TestReference(unittest.TestCase):

    def test_classification_label_match(self):
        self.assertTrue(check_reference("Text classification", "Positive.", "positive")["correct"])
        self.assertTrue(
            check_reference("Text classification", "The sentiment is Negative", "Negative")["correct"]
        )
        self.assertFalse(check_reference("Text classification", "Neutral", "Positive")["correct"])

    def test_extraction_field_match(self):
        reference = '{"name": "Maya", "date": "12 June"}'
        full = check_reference("Information extraction", '{"name": "maya", "date": "12 June"}', reference)
        partial = check_reference("Information extraction", '{"name": "Maya", "date": "13 June"}', reference)
        broken = check_reference("Information extraction", "Maya, 12 June", reference)

        self.assertTrue(full["correct"])
        self.assertEqual(partial["field_accuracy"], 0.5)
        self.assertFalse(broken["correct"])

    def test_extraction_with_non_json_reference(self):
        self.assertIsNone(check_reference("Information extraction", "{}", "Maya")["correct"])

    def test_other_categories_use_token_f1(self):
        result = check_reference("Summarization", "trees improve shade", "trees improve air quality")
        self.assertEqual(result["method"], "token_f1")
        self.assertGreater(result["f1"], 0)

    def test_token_f1_edges(self):
        self.assertEqual(token_f1("", "abc"), 0.0)
        self.assertEqual(token_f1("same words", "Same words!"), 1.0)


class TestValidateResponse(unittest.TestCase):

    def test_structured_output_answer_checked_against_reference(self):
        report = validate_response(
            "structured_output", "Text classification", "Classify: I love it",
            VALID_STRUCTURED, reference_answer="Positive",
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["json_valid"])
        self.assertTrue(report["checks"]["reference"]["correct"])

    def test_extraction_requires_json(self):
        report = validate_response(
            "zero_shot", "Information extraction", "Extract the name", "Maya",
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["json_valid"])

    def test_wrong_reference_fails(self):
        report = validate_response(
            "zero_shot", "Text classification", "Classify", "Negative", reference_answer="Positive",
        )
        self.assertFalse(report["passed"])
        self.assertIn("Response does not match the reference answer.", report["issues"])

    def test_plain_response_without_rules_passes(self):
        report = validate_response("zero_shot", "General question answering", "Explain ML", "ML is...")
        self.assertTrue(report["passed"])
        self.assertNotIn("json_valid", report["checks"])


if __name__ == "__main__":
    unittest.main()
