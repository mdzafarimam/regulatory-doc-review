"""LLM client factory shared by all regulatory-review reviewer agents."""

import os

from langchain_openai import AzureChatOpenAI

from common.config import (
    OPENAI_API_KEY,
    OPENAI_API_URL,
    OPENAI_API_VERSION,
    OPENAI_MODEL,
    LANGSMITH_API_KEY,
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
        # _llm = AzureChatOpenAI(
        #     azure_endpoint=OPENAI_API_URL,
        #     api_key=OPENAI_API_KEY,
        #     api_version=OPENAI_API_VERSION,
        #     azure_deployment=OPENAI_MODEL,
        #     temperature=0,
        # )
        _llm = ChatOpenAI(
        base_url="https://gateway.smith.langchain.com/azure/v1",
        api_key=LANGSMITH_API_KEY,  # the deployment's own key
        model=OPENAI_MODEL,                        # his Azure deployment name
        )
    return _llm


from langchain_openai import ChatOpenAI
