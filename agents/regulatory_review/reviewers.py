"""Reviewer agents, each focused on one aspect of regulatory document review."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.regulatory_review.llm import get_llm
from agents.regulatory_review.state import ReviewState


def prepare_document(state: ReviewState) -> dict:
    """Derive document text from direct input or the latest human message."""
    if state.get("document"):
        return {}

    for message in reversed(state.get("messages", [])):
        is_human = (
            isinstance(message, HumanMessage)
            or getattr(message, "type", None) == "human"
        )
        if is_human:
            content = (
                message.content
                if hasattr(message, "content")
                else message.get("content", "")
            )
            if isinstance(content, list):
                content = "\n".join(
                    block if isinstance(block, str) else block.get("text", "")
                    for block in content
                    if isinstance(block, str)
                    or isinstance(block, dict)
                    and block.get("type") == "text"
                )
            return {"document": content}

    return {"document": ""}


def validate_document(state: ReviewState) -> list[str]:
    """Route blank or instruction-only input away from the reviewers."""
    document = state.get("document", "").strip()
    instruction_only = re.fullmatch(
        r"(?:(?:can|could|would) you\s+)?(?:please\s+)?review\s+"
        r"(?:(?:the|this|my|a)\s+)?(?:regulatory\s+)?document"
        r"(?:\s+please)?[.!?]*",
        document,
        flags=re.IGNORECASE,
    )
    if not document or instruction_only:
        return ["request_document"]
    return ["completeness", "compliance", "clarity", "safety"]


def request_document(state: ReviewState) -> dict:
    """Ask for document content without producing a review report."""
    return {
        "final_report": "",
        "messages": [
            AIMessage(content="Please paste the regulatory document you want reviewed.")
        ],
    }


def _run_reviewer(reviewer_name: str, system_prompt: str, document: str) -> dict:
    """Invoke the LLM with a reviewer-specific system prompt and the document."""
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Regulatory document to review:\n\n{document}"),
    ]
    response = llm.invoke(messages)
    return {
        "reviews": [
            {
                "reviewer": reviewer_name,
                "findings": response.content,
            }
        ]
    }


def completeness_agent(state: ReviewState) -> dict:
    """Checks the document contains all expected regulatory sections."""
    system_prompt = (
        "You are a Regulatory Completeness Reviewer. Check whether the document "
        "contains all sections typically required in a regulatory submission "
        "(e.g., title/header, objectives, scope, methodology, risk assessment, "
        "safety data, references, approval/sign-off section). "
        "List any missing or incomplete sections. Be concise and use bullet points."
    )
    return _run_reviewer("Completeness Reviewer", system_prompt, state["document"])


def compliance_agent(state: ReviewState) -> dict:
    """Checks the document against general regulatory compliance guidelines."""
    system_prompt = (
        "You are a Regulatory Compliance Reviewer specializing in guidelines "
        "such as ICH, GxP, and FDA/EMA submission standards. Identify any "
        "statements, claims, or omissions that could violate regulatory "
        "compliance requirements (e.g., unsupported efficacy/safety claims, "
        "missing risk disclosures, inconsistent terminology). "
        "Be concise and use bullet points, citing the relevant guideline family "
        "where applicable."
    )
    return _run_reviewer("Compliance Reviewer", system_prompt, state["document"])


def clarity_agent(state: ReviewState) -> dict:
    """Checks the document for clarity, consistency, and language quality."""
    system_prompt = (
        "You are a Regulatory Document Language Reviewer. Review the document "
        "for clarity, ambiguity, inconsistent terminology, and readability "
        "issues that could confuse reviewers or regulators. "
        "Be concise and use bullet points."
    )
    return _run_reviewer("Clarity Reviewer", system_prompt, state["document"])


def safety_agent(state: ReviewState) -> dict:
    """Checks that safety-relevant information is adequately represented."""
    system_prompt = (
        "You are a Drug Safety Reviewer. Evaluate whether safety-relevant "
        "information (adverse events, risk mitigation, contraindications, "
        "monitoring requirements) is present, accurate, and sufficiently "
        "detailed. Flag any gaps or concerns. "
        "Be concise and use bullet points."
    )
    return _run_reviewer("Safety Reviewer", system_prompt, state["document"])


def supervisor_agent(state: ReviewState) -> dict:
    """Consolidates all reviewer findings into a single final report."""
    llm = get_llm()
    combined_findings = "\n\n".join(
        f"### {review['reviewer']}\n{review['findings']}" for review in state["reviews"]
    )
    system_prompt = (
        "You are the Regulatory Review Supervisor. You will be given findings "
        "from multiple specialist reviewers (completeness, compliance, clarity, "
        "safety) for the same regulatory document. "
        "If the findings establish that no document content was supplied, state "
        "that review could not be performed and ask for the document without "
        "issuing a verdict. Missing sections or incomplete substantive documents "
        "do not qualify for this exception. Otherwise, produce a single, "
        "consolidated review report with these sections: "
        "1) Executive Summary, 2) Key Issues (prioritized, most critical first), "
        "3) Section-by-section Recommendations, 4) Overall Verdict "
        "(Approve / Approve with changes / Reject), with justification."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Specialist reviewer findings:\n\n{combined_findings}"),
    ]
    response = llm.invoke(messages)
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)],
    }
