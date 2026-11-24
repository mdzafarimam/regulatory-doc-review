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
        "You are a Regulatory Completeness Reviewer. First identify the document "
        "type and scope (full document, synopsis, executive summary, or excerpt). "
        "Assess sections and detail appropriate to that type and scope: "
        "protocol synopses cover title/header, scope, objectives, methodology, "
        "risk assessment, safety assessments, references, and approval; "
        "prescribing information covers indications, dosage, contraindications, "
        "warnings, adverse reactions, interactions, specific populations, and "
        "clinical studies; PSUR summaries cover introduction, authorization "
        "status, safety actions, reference safety information changes, exposure, "
        "case presentation, signal evaluation, benefit-risk analysis, conclusion, "
        "references, and approval. Investigator-brochure excerpts need substantive "
        "nonclinical and clinical evidence and actionable safety guidance, even "
        "when abbreviated; placeholders are not evidence. For other types, use "
        "their applicable requirements. Recognize equivalent headings and content, "
        "including headers above numbered sections. Inspect the whole input before "
        "calling a section absent; quote the nearest relevant heading or state "
        "that none exists. Distinguish absent sections from present sections "
        "needing detail. Protocol-only sections do not apply to other formats, "
        "and summaries do not require full-document detail. State type and scope, "
        "then list supported gaps concisely with bullet points."
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
        "safety), alongside the source document. Identify its type and scope "
        "(including synopsis, summary, or excerpt), then verify every finding "
        "against the source and applicable requirements before using it. "
        "Discard contradicted or inapplicable findings; recognize equivalent "
        "headings and content, and distinguish absent sections from insufficient "
        "detail. Cite source text for key issues; for omissions, identify the "
        "relevant section or its absence without inventing quotations. "
        "Classify verified findings as minor (editorial), major (substantive but "
        "correctable), or critical (requires substantial reconstruction). "
        "Map severity to verdict: Approve for no substantive issues or minor "
        "issues only; Approve with changes for major, correctable deficiencies; "
        "Reject for critical deficiencies such as core evidence replaced by "
        "placeholders or pervasive promotional, unsupported safety/efficacy "
        "assertions. Correctable omissions, limited excerpt detail, and isolated "
        "claims needing substantiation do not by themselves justify Reject. "
        "Judge summaries at their stated scope, while flagging actionable gaps "
        "needed to finalize them. Base the verdict on verified severity, not "
        "the number of reviewer claims. "
        "Produce a single, consolidated review report with these sections: "
        "1) Executive Summary, 2) Key Issues (prioritized, most critical first), "
        "3) Section-by-section Recommendations, 4) Overall Verdict "
        "(Approve / Approve with changes / Reject), with justification."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Source document:\n\n{state['document']}\n\n"
                f"Specialist reviewer findings:\n\n{combined_findings}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)],
    }
