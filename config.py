"""Configuration and Gemini client setup for PromptBench."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"
DEFAULT_DATA_DIR = "experiments"

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    judge_model: str
    data_dir: str
    max_retries: int = 3
    request_timeout_seconds: float = 60.0


def _read_streamlit_secret(name: str) -> str | None:
    """Read a value from .streamlit/secrets.toml if Streamlit has one."""

    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        # No secrets file, or not running inside Streamlit.
        return None


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment or Streamlit secrets."""

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        key = _read_streamlit_secret("GEMINI_API_KEY")

    if key and key.strip():
        return key.strip()
    return None


def get_settings() -> Settings:
    """Collect PromptBench settings from the environment."""

    return Settings(
        api_key=get_api_key(),
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        judge_model=os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        data_dir=os.getenv("PROMPTBENCH_DATA_DIR", DEFAULT_DATA_DIR),
    )


def create_client(settings: Settings | None = None):
    """Create a Gemini client, failing clearly when no key is configured."""

    settings = settings or get_settings()

    if not settings.api_key:
        raise ConfigError(
            "No Gemini API key found. Set GEMINI_API_KEY in your "
            "environment, a .env file, or .streamlit/secrets.toml."
        )

    from google import genai

    return genai.Client(api_key=settings.api_key)


def check_connection(client=None, model: str | None = None) -> str:
    """Send a tiny request to Gemini and return its reply text."""

    settings = get_settings()
    client = client or create_client(settings)

    response = client.models.generate_content(
        model=model or settings.model,
        contents="Reply with the single word: ready",
    )
    return (response.text or "").strip()


if __name__ == "__main__":
    print(f"Gemini replied: {check_connection()}")
