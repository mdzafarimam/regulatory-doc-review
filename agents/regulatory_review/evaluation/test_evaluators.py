"""Unit tests for deterministic verdict scoring."""

import unittest
from types import SimpleNamespace

from agents.regulatory_review.evaluation.evaluators import (
    ALL_EVALUATORS,
    verdict_match_evaluator,
    verdict_present_evaluator,
)


class VerdictEvaluatorTests(unittest.TestCase):
    def test_matches_overall_verdict_only(self):
        for label in ("Approve", "Approve with changes", "Reject"):
            for heading in ("## Overall Verdict\n", "4) **Overall Verdict:** "):
                with self.subTest(label=label, heading=heading):
                    run = SimpleNamespace(
                        outputs={
                            "final_report": f"Executive Summary: Reject was considered.\n"
                            f"{heading}**{label}**\nJustification: verified findings."
                        }
                    )
                    example = SimpleNamespace(
                        outputs={"expected_verdict": label.lower()}
                    )
                    self.assertEqual(
                        verdict_match_evaluator(run, example)["score"], 1.0
                    )
                    self.assertEqual(
                        verdict_present_evaluator(run, example)["score"], 1.0
                    )

    def test_mismatch_missing_and_unrecognized_verdicts(self):
        for report in (
            "## Overall Verdict\nReject",
            "Executive Summary: Approve with changes",
            "## Overall Verdict\nUndecided; reviewers may approve later.",
            "## Overall Verdict\n\n## Appendix\nApprove with changes",
            "",
        ):
            with self.subTest(report=report):
                run = SimpleNamespace(outputs={"final_report": report})
                example = SimpleNamespace(
                    outputs={"expected_verdict": "approve with changes"}
                )
                self.assertEqual(verdict_match_evaluator(run, example)["score"], 0.0)

    def test_minor_changes_normalization(self):
        run = SimpleNamespace(
            outputs={"final_report": "Overall Verdict: APPROVE WITH MINOR CHANGES"}
        )
        for expected in ("approve", " Approve with minor changes "):
            with self.subTest(expected=expected):
                example = SimpleNamespace(outputs={"expected_verdict": expected})
                self.assertEqual(verdict_match_evaluator(run, example)["score"], 1.0)

    def test_missing_outputs_do_not_match(self):
        run = SimpleNamespace(outputs=None)
        example = SimpleNamespace(outputs=None)
        self.assertEqual(verdict_match_evaluator(run, example)["score"], 0.0)
        self.assertEqual(verdict_present_evaluator(run, example)["score"], 0.0)

    def test_match_is_registered_for_runner(self):
        self.assertIn(verdict_match_evaluator, ALL_EVALUATORS)


if __name__ == "__main__":
    unittest.main()
