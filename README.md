# PromptBench
An intelligent platform for testing, evaluating, and optimizing prompting techniques.

PromptBench turns one task into several prompt variants (zero-shot, few-shot,
role prompting and more), runs each variant on Gemini, and scores the
responses with an LLM judge plus rule-based checks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your key in .env
```

The Gemini API key is read from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the
environment, a `.env` file, or `.streamlit/secrets.toml`. All of these are
gitignored. Never commit a key.

Check the key works:

```bash
python config.py
```

## Run

```bash
streamlit run app.py
```

Saved experiments are written as JSON to `experiments/` (override with
`PROMPTBENCH_DATA_DIR`). Override models with `GEMINI_MODEL` and
`GEMINI_JUDGE_MODEL`.

## Test

```bash
python -m pytest -q
```

Tests mock every Gemini call, so they need no API key.

## Modules

| File | Role |
|---|---|
| `app.py` | Streamlit UI |
| `prompt_generator.py` | Builds prompt variants per technique and category |
| `config.py` | Settings, API key loading and the Gemini client |
| `experiment_runner.py` | Runs each variant, capturing response, latency, tokens and errors, with retries for rate limits |
| `evaluator.py` | LLM judge scoring relevance, completeness, clarity and instruction following (1 to 10) |
| `validators.py` | JSON validity, length constraints and reference-answer checks |
| `storage.py` | Saves and loads experiments with IDs, timestamps and metadata |
| `backend.py` | Runs, validates, judges and saves a full experiment |
| `results_dashboard.py` | Results table, metrics and charts |

## Result format

Each result row has `technique`, `prompt`, `response`, `latency_seconds`,
`input_tokens`, `output_tokens`, `total_tokens`, `evaluation_score`, plus
`status` (`success`, `error`, `rate_limited` or `empty`), `error`, `attempts`,
`judge_scores`, `judge_feedback` and `validation`.
