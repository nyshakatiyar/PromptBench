
"""Generate prompt variants for PromptBench."""

from typing import Iterable


TECHNIQUES = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "role_prompting": "Role prompting",
    "constraint_based": "Constraint-based",
    "structured_output": "Structured output",
    "systematic_reasoning": "Systematic reasoning",
}


FEW_SHOT_EXAMPLES = """
Examples of how to respond to tasks:

Task: Explain photosynthesis to a beginner.
Response: Photosynthesis is the process plants use to convert
sunlight, water, and carbon dioxide into sugars and oxygen.

Task: Explain why exercise is beneficial.
Response: Exercise supports physical health, improves fitness,
and can benefit mood and general well-being.

Use the examples as guidance for clear, beginner-friendly
explanations. Adapt your response to the actual task.
"""


def generate_prompts(
    task: str,
    techniques: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return selected prompt variants keyed by technique ID."""

    if not isinstance(task, str) or not task.strip():
        raise ValueError("Please enter a task or prompt.")

    selected = list(techniques) if techniques is not None else list(TECHNIQUES)

    if not selected:
        raise ValueError("Select at least one prompting technique.")

    unknown = set(selected) - set(TECHNIQUES)
    if unknown:
        raise ValueError(
            f"Unknown techniques: {', '.join(sorted(unknown))}"
        )

    task = task.strip()
    prompts = {}

    for technique in selected:
        if technique == "zero_shot":
            prompt = task

        elif technique == "few_shot":
            prompt = (
                f"{FEW_SHOT_EXAMPLES}\n\n"
                f"Now complete this task:\n{task}"
            )

        elif technique == "role_prompting":
            prompt = (
                "Act as a knowledgeable subject-matter expert "
                "who can explain ideas clearly to the intended audience.\n\n"
                f"Task:\n{task}"
            )

        elif technique == "constraint_based":
            prompt = (
                f"Complete the following task:\n{task}\n\n"
                "Requirements:\n"
                "- Be clear and relevant.\n"
                "- Avoid unnecessary repetition.\n"
                "- Follow any specific constraints in the original task.\n"
                "- If no length or format is specified, respond concisely."
            )

        elif technique == "structured_output":
            prompt = (
                f"Complete the following task:\n{task}\n\n"
                "Return a valid JSON object with these keys:\n"
                '- "answer": the direct response to the task\n'
                '- "key_points": an array of important points\n'
                '- "limitations": an array of relevant caveats, '
                "or an empty array if none apply\n\n"
                "Return JSON only, without Markdown code fences."
            )

        elif technique == "systematic_reasoning":
            prompt = (
                f"Complete the following task carefully:\n{task}\n\n"
                "Consider the relevant facts and requirements before "
                "answering. Check that the answer addresses the task "
                "and follows its constraints. Provide the final answer "
                "with a concise explanation or rationale, without "
                "revealing private internal reasoning."
            )

        prompts[technique] = prompt

    return prompts
