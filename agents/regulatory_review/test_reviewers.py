"""Review flow wiring tests and credential-dependent model behavior tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.regulatory_review.evaluation.dataset import EXAMPLES
from agents.regulatory_review.evaluation.evaluators import _extract_verdict
from agents.regulatory_review.evaluation.runner import target
from agents.regulatory_review.reviewers import supervisor_agent
from common.config import LANGSMITH_API_KEY, OPENAI_API_KEY, OPENAI_API_URL


class ReviewFlowTests(unittest.TestCase):
    def test_supervisor_can_read_source_and_findings(self):
        document = EXAMPLES[1]["inputs"]["document"]
        findings = "Risk Assessment section is absent."

        def verify_source(messages):
            content = messages[-1].content
            self.assertIn(document, content)
            self.assertIn(findings, content)
            return SimpleNamespace(content="Overall Verdict: Approve")

        llm = SimpleNamespace(invoke=verify_source)
        with patch("agents.regulatory_review.reviewers.get_llm", return_value=llm):
            result = supervisor_agent(
                {
                    "document": document,
                    "reviews": [
                        {"reviewer": "Completeness Reviewer", "findings": findings}
                    ],
                }
            )
        self.assertEqual(result["final_report"], "Overall Verdict: Approve")
        self.assertEqual(result["messages"][0].content, result["final_report"])

    def test_runner_passes_each_golden_document_through_full_graph(self):
        for example in EXAMPLES:
            with self.subTest(expected=example["outputs"]["expected_verdict"]):
                document = example["inputs"]["document"]

                def review(messages, document=document):
                    self.assertIn(document, messages[-1].content)
                    return SimpleNamespace(content="Overall Verdict: Approve")

                with patch(
                    "agents.regulatory_review.reviewers.get_llm",
                    return_value=SimpleNamespace(invoke=review),
                ):
                    result = target(example["inputs"])
                self.assertEqual(len(result["reviews"]), 4)
                self.assertEqual(result["final_report"], "Overall Verdict: Approve")


@unittest.skipUnless(
    OPENAI_API_KEY and OPENAI_API_URL and LANGSMITH_API_KEY,
    "Live review tests require configured model credentials and endpoint.",
)
class LiveReviewTests(unittest.TestCase):
    def test_golden_verdicts_through_runner(self):
        for example in EXAMPLES:
            with self.subTest(expected=example["outputs"]["expected_verdict"]):
                result = target(example["inputs"])
                self.assertEqual(
                    _extract_verdict(result["final_report"]),
                    example["outputs"]["expected_verdict"],
                    result["final_report"],
                )
                completeness = next(
                    review["findings"]
                    for review in result["reviews"]
                    if review["reviewer"] == "Completeness Reviewer"
                ).lower()
                if example is EXAMPLES[4]:
                    self.assertNotRegex(
                        completeness,
                        r"(?:missing|absent|lacks?)\s+(?:a\s+)?(?:title|header|objectives|scope|methodology)",
                    )
                elif example is EXAMPLES[3]:
                    self.assertNotRegex(
                        completeness,
                        r"(?:missing|absent|lacks?)\s+(?:objectives|scope)",
                    )
                elif example is EXAMPLES[1]:
                    self.assertNotRegex(
                        completeness,
                        r"(?:missing|absent|lacks?)\s+(?:a\s+)?risk assessment",
                    )

    def test_contradicted_and_inapplicable_findings_do_not_change_verdict(self):
        cases = (
            (1, "Critical: Risk Assessment section is missing; add it. Reject."),
            (
                4,
                "Critical: Header, objectives, scope and methodology are missing. Reject.",
            ),
        )
        for index, findings in cases:
            with self.subTest(index=index):
                result = supervisor_agent(
                    {
                        "document": EXAMPLES[index]["inputs"]["document"],
                        "reviews": [
                            {"reviewer": "Completeness Reviewer", "findings": findings}
                        ],
                    }
                )
                self.assertEqual(
                    _extract_verdict(result["final_report"]), "approve", result
                )

    def test_verified_severity_controls_verdict(self):
        cases = (
            (4, "Minor: expand CCDS on first use.", "approve"),
            (
                3,
                "Major: renal dosing table and actionable monitoring thresholds need completion.",
                "approve with changes",
            ),
            (
                0,
                "Major: missing scope, risk assessment, and approval; substantiate tolerability claims.",
                "approve with changes",
            ),
            (
                2,
                "Critical: core nonclinical and clinical evidence is replaced by placeholders, with pervasive unsupported safety claims.",
                "reject",
            ),
        )
        for index, findings, expected in cases:
            with self.subTest(expected=expected, index=index):
                result = supervisor_agent(
                    {
                        "document": EXAMPLES[index]["inputs"]["document"],
                        "reviews": [
                            {"reviewer": "Specialist Reviewer", "findings": findings}
                        ],
                    }
                )
                self.assertEqual(
                    _extract_verdict(result["final_report"]), expected, result
                )


if __name__ == "__main__":
    unittest.main()
