"""LLM client factory shared by all regulatory-review reviewer agents."""

from langchain_openai import AzureChatOpenAI

from common.config import (
    OPENAI_API_KEY,
    OPENAI_API_URL,
    OPENAI_API_VERSION,
    OPENAI_MODEL,
)

_llm: AzureChatOpenAI | None = None


def get_llm() -> AzureChatOpenAI:
    """Return a shared, lazily-created AzureChatOpenAI client."""
    global _llm
    if _llm is None:
        if not OPENAI_API_KEY or not OPENAI_API_URL:
            raise ValueError(
                "OPENAI_API_KEY and OPENAI_API_URL must be set in the .env file."
            )
        _llm = AzureChatOpenAI(
            azure_endpoint=OPENAI_API_URL,
            api_key=OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
            azure_deployment=OPENAI_MODEL,
            temperature=0,
        )
    return _llm
