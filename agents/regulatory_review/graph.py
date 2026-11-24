"""Builds and runs the multi-agent regulatory document review graph.

Usage:
    python agents/regulatory_review/graph.py path/to/document.txt

Environment-aware graph factories (used by langgraph.json for LangGraph
Platform / LangSmith deployment) are provided below: `make_graph` builds the
graph for whichever environment is configured via the APP_ENV / ENVIRONMENT
env var (defaulting to "dev"), and `make_staging_graph` / `make_prod_graph`
are explicit variants for the "agent_staging" / "agent_prod" graph entries.
"""

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langgraph.graph import END, START, StateGraph

from agents.regulatory_review.reviewers import (
    clarity_agent,
    completeness_agent,
    compliance_agent,
    prepare_document,
    safety_agent,
    supervisor_agent,
)
from agents.regulatory_review.state import ReviewState


def build_review_graph():
    """Build the LangGraph StateGraph for the multi-agent review system.

    Flow:
        START -> [completeness, compliance, clarity, safety] (run in parallel)
              -> supervisor (consolidates findings into final_report)
              -> END
    """
    graph = StateGraph(ReviewState)

    graph.add_node("prepare_document", prepare_document)
    graph.add_node("completeness", completeness_agent)
    graph.add_node("compliance", compliance_agent)
    graph.add_node("clarity", clarity_agent)
    graph.add_node("safety", safety_agent)
    graph.add_node("supervisor", supervisor_agent)

    graph.add_edge(START, "prepare_document")

    # Fan-out: all four specialist reviewers run in parallel after the
    # document has been resolved from either direct input or chat messages.
    for reviewer_node in ("completeness", "compliance", "clarity", "safety"):
        graph.add_edge("prepare_document", reviewer_node)
        graph.add_edge(reviewer_node, "supervisor")

    graph.add_edge("supervisor", END)

    return graph.compile()


def review_document(document_text: str) -> ReviewState:
    """Run the full multi-agent review over the given document text."""
    app = build_review_graph()
    result = app.invoke(
        {"messages": [], "document": document_text, "reviews": [], "final_report": ""}
    )
    return result


def _current_environment() -> str:
    return (os.environ.get("APP_ENV") or os.environ.get("ENVIRONMENT") or "dev").lower()


def make_graph(config: dict | None = None):
    """Factory used by langgraph.json's "agent" graph entry.

    Builds the graph for whichever environment is active (APP_ENV /
    ENVIRONMENT, default "dev"). LangGraph Platform calls this with an
    optional `config` dict; it isn't needed for this stateless graph.
    """
    return build_review_graph()


def make_dev_graph(config: dict | None = None):
    """Factory for the "agent_dev" graph entry — forces the dev environment."""
    os.environ.setdefault("APP_ENV", "dev")
    return build_review_graph()


def make_staging_graph(config: dict | None = None):
    """Factory for the "agent_staging" graph entry — forces the staging environment."""
    os.environ.setdefault("APP_ENV", "staging")
    return build_review_graph()


def make_prod_graph(config: dict | None = None):
    """Factory for the "agent_prod" graph entry — forces the prod environment."""
    os.environ.setdefault("APP_ENV", "prod")
    return build_review_graph()


# Module-level compiled graph instance for ad-hoc/local use
# (e.g. `from agents.regulatory_review.graph import graph`).
graph = build_review_graph()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        doc_path = Path(sys.argv[1])
    else:
        doc_path = (
            Path(__file__).resolve().parent.parent.parent
            / "sample_documents"
            / "sample_clinical_protocol.txt"
        )

    text = doc_path.read_text(encoding="utf-8")
    final_state = review_document(text)

    print("=" * 80)
    print("INDIVIDUAL REVIEWER FINDINGS")
    print("=" * 80)
    for review in final_state["reviews"]:
        print(f"\n--- {review['reviewer']} ---\n{review['findings']}")

    print("\n" + "=" * 80)
    print("FINAL CONSOLIDATED REPORT")
    print("=" * 80)
    print(final_state["final_report"])
