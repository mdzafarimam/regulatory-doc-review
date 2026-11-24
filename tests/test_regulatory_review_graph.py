import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.regulatory_review.graph import build_review_graph, review_document


class ReviewGraphTests(unittest.TestCase):
    def test_missing_document_requests_content_without_model_calls(self):
        inputs = [
            {},
            {"document": ""},
            {"document": " \n\t"},
            {"messages": [HumanMessage(content="")]},
            {"messages": [HumanMessage(content="review the document")]},
            {
                "messages": [
                    HumanMessage(content="Please review the regulatory document.")
                ]
            },
            {"messages": [HumanMessage(content=[{"type": "text", "text": " \n"}])]},
            {
                "messages": [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "review the"},
                            {"type": "text", "text": "regulatory document"},
                        ]
                    )
                ]
            },
            {"messages": [HumanMessage(content=["review the document"])]},
        ]
        with patch("agents.regulatory_review.reviewers.get_llm") as get_llm:
            for state in inputs:
                with self.subTest(state=state):
                    result = build_review_graph().invoke(
                        {**state, "final_report": "old report"}
                    )
                    self.assertEqual(result["final_report"], "")
                    self.assertEqual(result.get("reviews", []), [])
                    response = result["messages"][-1]
                    self.assertIsInstance(response, AIMessage)
                    self.assertIn("paste the regulatory document", response.content)
                    self.assertNotIn("Overall Verdict", response.content)
            get_llm.assert_not_called()

    def test_substantive_documents_complete_all_reviews_and_supervision(self):
        documents = [
            "Adverse events must be reported within 24 hours.",
            "Review the document\n\nSafety: Stop treatment if a serious adverse event occurs.",
            "review the regulatory document: Monitor blood pressure daily.",
        ]
        for document in documents:
            inputs = [
                {"document": document},
                {"messages": [HumanMessage(content=document)]},
                {
                    "messages": [
                        HumanMessage(
                            content=[
                                {"type": "text", "text": document},
                            ]
                        )
                    ]
                },
            ]
            for state in inputs:
                with self.subTest(state=state):
                    llm = Mock()
                    llm.invoke.return_value = AIMessage(content="Reviewed")
                    with patch(
                        "agents.regulatory_review.reviewers.get_llm", return_value=llm
                    ):
                        result = build_review_graph().invoke(state)
                    self.assertEqual(result["document"], document)
                    self.assertEqual(
                        {review["reviewer"] for review in result["reviews"]},
                        {
                            "Completeness Reviewer",
                            "Compliance Reviewer",
                            "Clarity Reviewer",
                            "Safety Reviewer",
                        },
                    )
                    self.assertEqual(result["final_report"], "Reviewed")
                    self.assertEqual(result["messages"][-1].content, "Reviewed")
                    self.assertEqual(llm.invoke.call_count, 5)
                    supervisor_input = llm.invoke.call_args.args[0][-1].content
                    for review in result["reviews"]:
                        self.assertIn(review["reviewer"], supervisor_input)

    def test_programmatic_empty_document_ends_without_review(self):
        with patch("agents.regulatory_review.reviewers.get_llm") as get_llm:
            result = review_document("")
        self.assertEqual(result["final_report"], "")
        self.assertIn("paste the regulatory document", result["messages"][-1].content)
        get_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
