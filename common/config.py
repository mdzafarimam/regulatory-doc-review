"""Shared configuration loaded once from .env for every agent and the client."""

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    # On corporate networks with TLS-inspecting proxies, this extends the
    # certifi CA bundle with certs from the OS (Windows) trust store, so
    # outbound HTTPS calls (OpenAI, LangSmith, etc.) can verify certificates.
    import certifi_win32  # noqa: F401
except (ImportError, KeyError):
    # Not on Windows (e.g. a Linux Agent Server), so there is no Windows trust store.
    pass

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# If your network uses a TLS-inspecting proxy (common on corporate networks),
# point this at the corporate root CA bundle (.pem/.crt) so outbound HTTPS
# calls (OpenAI, LangSmith, etc.) can verify certificates correctly.
_CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
if _CA_BUNDLE:
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA_BUNDLE)
    os.environ.setdefault("SSL_CERT_FILE", _CA_BUNDLE)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL") or None
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview")

# LangSmith (tracing / evaluation framework). Supports both the newer
# LANGSMITH_* names and the legacy LANGCHAIN_* names (the SDK reads the
# latter directly from os.environ, so either naming works).
LANGSMITH_API_KEY = (
    os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY") or ""
)
LANGSMITH_PROJECT = os.environ.get(
    "LANGSMITH_PROJECT", os.environ.get("LANGCHAIN_PROJECT", "regulatory-review")
)
LANGSMITH_TRACING = os.environ.get(
    "LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", "true")
)
LANGSMITH_ENDPOINT = os.environ.get(
    "LANGSMITH_ENDPOINT",
    os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
)

# Wire LangSmith env vars (the SDK reads these directly from os.environ).
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    os.environ.setdefault("LANGSMITH_TRACING", LANGSMITH_TRACING)
    os.environ.setdefault("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)

MS_AGENT_HOST = os.environ.get("MS_AGENT_HOST", "127.0.0.1")
MS_AGENT_PORT = int(os.environ.get("MS_AGENT_PORT", "9001"))

LANGGRAPH_AGENT_HOST = os.environ.get("LANGGRAPH_AGENT_HOST", "127.0.0.1")
LANGGRAPH_AGENT_PORT = int(os.environ.get("LANGGRAPH_AGENT_PORT", "9002"))

SK_AGENT_HOST = os.environ.get("SK_AGENT_HOST", "127.0.0.1")
SK_AGENT_PORT = int(os.environ.get("SK_AGENT_PORT", "9003"))
