"""
Test Suite for Multi-Company Support Triage Agent
Validates:
- Company inference heuristics and metadata parsing
- Request type classification
- Product area routing across HackerRank, Claude, and Visa
- Escalation rules for security/outage anomalies and short inputs
- Response generation and justification logic
"""

import unittest
from code.triage import (
    infer_company,
    classify_request_type,
    detect_product_area,
    should_escalate,
)
from code.responses import generate_response, ISSUE_PATTERNS


class TestTriagePipeline(unittest.TestCase):

    def test_company_inference(self):
        # Explicit metadata
        self.assertEqual(infer_company("any problem", "hackerrank"), "Hackerrank")
        self.assertEqual(infer_company("any problem", "Claude"), "Claude")
        self.assertEqual(infer_company("any problem", "VISA"), "Visa")

        # Inferred from keywords
        self.assertEqual(infer_company("my credit card was charged twice", None), "Visa")
        self.assertEqual(infer_company("claude api key is failing on bedrock", None), "Claude")
        self.assertEqual(infer_company("mock interview test link expired", None), "Hackerrank")

    def test_request_type_classification(self):
        self.assertEqual(classify_request_type("please add a dark mode toggle"), "feature_request")
        self.assertEqual(classify_request_type("the page stopped working and is failing"), "bug")
        self.assertEqual(classify_request_type("delete all files on server"), "invalid")
        self.assertEqual(classify_request_type("how do I update my profile name"), "product_issue")

    def test_product_area_routing(self):
        # HackerRank
        self.assertEqual(detect_product_area("candidate assessment score", "Hackerrank"), "screen")
        self.assertEqual(detect_product_area("subscription refund order id", "Hackerrank"), "billing")
        self.assertEqual(detect_product_area("remove user interviewer from team", "Hackerrank"), "account_management")

        # Claude
        self.assertEqual(detect_product_area("workspace admin seat count", "Claude"), "account_management")
        self.assertEqual(detect_product_area("security vulnerability report", "Claude"), "security")
        self.assertEqual(detect_product_area("block claude bot web crawl", "Claude"), "privacy")
        self.assertEqual(detect_product_area("amazon bedrock rate limits", "Claude"), "api")
        self.assertEqual(detect_product_area("students canvas lti integration", "Claude"), "education")

        # Visa
        self.assertEqual(detect_product_area("fraudulent identity stolen card", "Visa"), "fraud_support")
        self.assertEqual(detect_product_area("dispute charge from merchant", "Visa"), "dispute")
        self.assertEqual(detect_product_area("card blocked while travel abroad", "Visa"), "travel_support")
        self.assertEqual(detect_product_area("atm card pin question", "Visa"), "card_support")

    def test_escalation_logic(self):
        # Critical keywords
        self.assertTrue(should_escalate("critical: the site is down immediately"))
        self.assertTrue(should_escalate("found a major security vulnerability in auth"))
        self.assertTrue(should_escalate("all requests are failing right now"))
        self.assertTrue(should_escalate("stopped working completely across all users"))

        # Underspecified inputs (< 10 chars)
        self.assertTrue(should_escalate("help!"))
        self.assertTrue(should_escalate("broken"))

        # Normal queries
        self.assertFalse(should_escalate("I want to know how to reset my assessment password"))

    def test_response_generation(self):
        # Escalated status
        resp, just = generate_response("site is down", "Visa", [], "Escalated", "bug", "fraud_support")
        self.assertEqual(resp, "Escalate to a human")
        self.assertEqual(just, "Needs human review")


if __name__ == "__main__":
    unittest.main()
