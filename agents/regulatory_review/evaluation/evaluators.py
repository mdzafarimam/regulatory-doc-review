"""LangSmith evaluators (LLM-as-judge) for the regulatory review graph.

Each evaluator function follows the LangSmith `evaluate()` signature:
    def evaluator(run, example) -> dict | EvaluationResult

`run.outputs` is the dict returned by the target function (see runner.py).
`example.outputs` holds the reference data from the dataset.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.regulatory_review.llm import get_llm


class _JudgeScore(BaseModel):
    score: float = Field(description="Score between 0.0 (worst) and 1.0 (best).")
    reasoning: str = Field(description="Brief justification for the score.")


def _llm_judge(key: str, system_prompt: str, user_prompt: str) -> dict:
    """Call the LLM with a structured-output judge prompt and return a result dict."""
    llm = get_llm().with_structured_output(_JudgeScore)
    result: _JudgeScore = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return {
        "key": key,
        "score": result.score,
        "comment": result.reasoning,
    }


def coverage_evaluator(run, example) -> dict:
    """Judges whether the final report covers the issues noted in the reference."""
    final_report = (run.outputs or {}).get("final_report", "")
    reference_notes = (example.outputs or {}).get("reference_notes", "")

    system_prompt = (
        "You are an expert evaluator grading a regulatory document review report. "
        "Given a set of reference issues that a thorough review should surface, "
        "score how well the candidate report covers those issues. "
        "1.0 = all reference issues are substantively addressed. "
        "0.0 = none of the reference issues are addressed. "
        "Partial coverage should get a proportional score."
    )
    user_prompt = (
        f"Reference issues the review should cover:\n{reference_notes}\n\n"
        f"Candidate final report to grade:\n{final_report}"
    )
    return _llm_judge("coverage", system_prompt, user_prompt)


def groundedness_evaluator(run, example) -> dict:
    """Judges whether the report's claims are grounded in the source document."""
    inputs = run.inputs if hasattr(run, "inputs") else {}
    document = (inputs or {}).get("document", "") if isinstance(inputs, dict) else ""
    if not document:
        document = (example.inputs or {}).get("document", "")
    final_report = (run.outputs or {}).get("final_report", "")

    system_prompt = (
        "You are an expert evaluator checking for hallucination in a regulatory "
        "document review report. Score how well the report's claims are grounded "
        "in the actual source document content (not invented facts). "
        "1.0 = fully grounded, no fabricated claims. "
        "0.0 = report references content not present in the source document."
    )
    user_prompt = (
        f"Source document:\n{document}\n\n"
        f"Review report to check for grounding:\n{final_report}"
    )
    return _llm_judge("groundedness", system_prompt, user_prompt)


def structure_evaluator(run, example) -> dict:
    """Judges whether the final report follows the required structure/format."""
    final_report = (run.outputs or {}).get("final_report", "")

    required_sections = [
        "Executive Summary",
        "Key Issues",
        "Section-by-section Recommendations",
        "Overall Verdict",
    ]
    found = [s for s in required_sections if s.lower() in final_report.lower()]
    score = len(found) / len(required_sections)
    missing = [s for s in required_sections if s not in found]
    comment = (
        "All required sections present."
        if not missing
        else f"Missing sections: {', '.join(missing)}"
    )
    return {"key": "structure", "score": score, "comment": comment}


def verdict_present_evaluator(run, example) -> dict:
    """Checks that the final report contains one of the allowed verdict labels."""
    final_report = (run.outputs or {}).get("final_report", "").lower()
    allowed = ["approve with changes", "approve", "reject"]
    present = any(v in final_report for v in allowed)
    return {
        "key": "verdict_present",
        "score": 1.0 if present else 0.0,
        "comment": "Verdict label found." if present else "No recognizable verdict label found.",
    }


ALL_EVALUATORS = [
    coverage_evaluator,
    groundedness_evaluator,
    structure_evaluator,
    verdict_present_evaluator,
]
