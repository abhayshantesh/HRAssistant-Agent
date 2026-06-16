"""
Central configuration for HRAssistant-Agent.

All settings are read from environment variables (loaded from a .env file in
local development) with sensible defaults. The only required variable is
OPENROUTER_API_KEY.

This module deliberately does NOT import streamlit so it can be used from
scripts and tests. When running on Streamlit Cloud, secrets defined in
.streamlit/secrets.toml are exposed as environment variables automatically,
so os.getenv() picks them up without a hard streamlit dependency here.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _get_api_key() -> str | None:
    """Read the OpenRouter API key from env, falling back to Streamlit secrets.

    The Streamlit import is done lazily and defensively so importing config
    never fails outside a Streamlit runtime.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        return None


# --- LLM / OpenRouter ---------------------------------------------------------
OPENROUTER_API_KEY = _get_api_key()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Models are tried in order; the first that responds successfully is used.
# All are free tiers on OpenRouter (suffix :free). Override with a single
# model via OPENROUTER_MODEL if desired.
DEFAULT_MODELS = [
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-32b",
]
_model_override = os.getenv("OPENROUTER_MODEL")
MODELS = [_model_override] if _model_override else DEFAULT_MODELS

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# --- RAG ----------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Paths --------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMPLOYEE_DATA_PATH = os.getenv("EMPLOYEE_DATA_PATH", os.path.join(DATA_DIR, "employee_data.csv"))
POLICIES_DIR = os.getenv("POLICIES_DIR", os.path.join(DATA_DIR, "policies"))

# --- UI -----------------------------------------------------------------------
PAGE_TITLE = "HR Assistant Agent"
PAGE_ICON = "💼"
