"""Environment variable loading. Everything has a sensible default so the app
runs with zero configuration (no API key needed: classical NLP fallbacks)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

APP_NAME = "LitLens"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BACKEND_DIR / 'litlens.db').as_posix()}")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

HTTP_TIMEOUT = 30          # seconds, per external call
HTTP_RETRIES = 3
MAX_PDF_PAGES = 12         # only the first pages are needed for summaries/extraction
FRONTEND_DIR = ROOT_DIR / "frontend"


def llm_enabled() -> bool:
    return bool(LLM_API_KEY) and LLM_PROVIDER == "anthropic"
