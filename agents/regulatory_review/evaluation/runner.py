"""Run a LangSmith evaluation over the regulatory document review graph.

This script:
1. Creates (or reuses) a LangSmith dataset from `dataset.py`'s examples.
2. Defines a `target` function that invokes the review graph.
3. Runs `langsmith.evaluate()` with the evaluators from `evaluators.py`.

Requires LANGSMITH_API_KEY to be set in .env (see common/config.py).

Usage:
    python agents/regulatory_review/evaluation/runner.py
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langsmith import Client

from common.config import LANGSMITH_API_KEY
from agents.regulatory_review.evaluation.dataset import get_dataset_examples
from agents.regulatory_review.evaluation.evaluators import ALL_EVALUATORS
from agents.regulatory_review.graph import build_review_graph

DATASET_NAME = "regulatory-review-eval"


def ensure_dataset(client: Client, sync: bool = True) -> str:
    """Create the LangSmith dataset if it doesn't exist yet, and populate/sync it.

    If the dataset already exists and `sync` is True, any example in
    `dataset.py` whose input document isn't already present (matched by the
    `document` text) is added, so re-running this after editing the golden
    dataset keeps LangSmith up to date without creating duplicates.
    """
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        if sync:
            existing_docs = {
                ex.inputs.get("document")
                for ex in client.list_examples(dataset_id=dataset.id)
            }
            for example in get_dataset_examples():
                if example["inputs"].get("document") not in existing_docs:
                    client.create_example(
                        inputs=example["inputs"],
                        outputs=example["outputs"],
                        dataset_id=dataset.id,
                    )
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Golden evaluation dataset for the regulatory document review "
                "multi-agent graph: regulatory documents paired with "
                "expert-authored reference notes per reviewer dimension "
                "(completeness, compliance, clarity, safety) and an expected "
                "verdict, used by LLM-as-judge evaluators."
            ),
        )
        for example in get_dataset_examples():
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id,
            )
    return dataset.name


def upload_dataset() -> str:
    """Create/sync the golden dataset in LangSmith without running an evaluation.

    Usage:
        python -m agents.regulatory_review.evaluation.runner --upload-only
    """
    if not LANGSMITH_API_KEY:
        raise ValueError(
            "LANGSMITH_API_KEY must be set in the .env file to upload the dataset."
        )
    client = Client()
    name = ensure_dataset(client)
    print(f"Golden dataset '{name}' is up to date in LangSmith "
          f"({len(get_dataset_examples())} example(s) defined locally).")
    return name


def target(inputs: dict) -> dict:
    """LangSmith target function: runs the compiled graph for one dataset example."""
    app = build_review_graph()
    result = app.invoke(
        {"document": inputs["document"], "reviews": [], "final_report": ""}
    )
    return {
        "final_report": result["final_report"],
        "reviews": result["reviews"],
    }


def main() -> None:
    if not LANGSMITH_API_KEY:
        raise ValueError(
            "LANGSMITH_API_KEY must be set in the .env file to run evaluations."
        )

    if "--upload-only" in sys.argv:
        upload_dataset()
        return

    client = Client()
    dataset_name = ensure_dataset(client)

    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix="regulatory-review",
        description="Automated evaluation of the regulatory review multi-agent graph.",
    )

    print("Evaluation complete. View results in the LangSmith UI.")
    print(results)


if __name__ == "__main__":
    main()
