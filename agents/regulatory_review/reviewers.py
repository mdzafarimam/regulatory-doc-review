"""Reviewer agents, each focused on one aspect of regulatory document review."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.regulatory_review.llm import get_llm
from agents.regulatory_review.state import ReviewState


def prepare_document(state: ReviewState) -> dict:
    """Derive the `document` field from the chat history.

    Lets this graph be driven by LangGraph Studio's Chat UI: the content of
    the latest human message is treated as the regulatory document text to
    review. If `document` was already supplied directly (e.g. programmatic
    invocation), it is left untouched.
    """
    if state.get("document"):
        return {}

    for message in reversed(state.get("messages", [])):
        is_human = isinstance(message, HumanMessage) or getattr(message, "type", None) == "human"
        if is_human:
            content = message.content if hasattr(message, "content") else message.get("content", "")
            return {"document": content}

    return {"document": ""}


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
        "Produce a single, consolidated review report with these sections: "
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
