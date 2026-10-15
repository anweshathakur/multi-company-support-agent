from typing import Any
import pandas as pd

try:
    from config import MIN_TICKET_LENGTH_FOR_ESCALATION
except ImportError:
    from code.config import MIN_TICKET_LENGTH_FOR_ESCALATION


def infer_company(text: str, provided_company: Any) -> str:
    """
    Determine the company context for a ticket using metadata or keyword cues.
    """
    if pd.notna(provided_company) and str(provided_company).strip() not in ["", "None", "nan"]:
        return str(provided_company).strip().capitalize()

    t = text.lower()

    if any(k in t for k in ["visa", "card", "merchant", "charge", "fraud", "cash", "travel"]):
        return "Visa"
    elif any(k in t for k in ["claude", "anthropic", "bedrock", "workspace", "api key", "lti"]):
        return "Claude"
    else:
        return "Hackerrank"


def classify_request_type(text: str) -> str:
    """
    Classify the incoming ticket into: feature_request, bug, invalid, or product_issue.
    """
    t = text.lower()

    if any(x in t for x in ["feature request", "please add", "can you add"]):
        return "feature_request"

    if any(x in t for x in ["down", "not working", "stopped", "failing", "error"]):
        return "bug"

    if any(x in t for x in ["delete all files", "iron man actor", "random question"]):
        return "invalid"

    return "product_issue"


def detect_product_area(text: str, company: str) -> str:
    """
    Route the ticket into company-specific product areas.
    """
    t = text.lower()

    if company == "Hackerrank":
        if any(k in t for k in ["assessment", "test", "candidate", "reschedule", "score"]):
            return "screen"
        if any(k in t for k in ["subscription", "payment", "refund", "order id", "charged"]):
            return "billing"
        if any(k in t for k in ["employee", "interviewer", "remove user"]):
            return "account_management"
        return "screen"

    elif company == "Claude":
        if any(k in t for k in ["workspace", "seat", "admin"]):
            return "account_management"
        if any(k in t for k in ["security vulnerability", "bug bounty"]):
            return "security"
        if any(k in t for k in ["crawl", "website", "data"]):
            return "privacy"
        if any(k in t for k in ["bedrock", "aws"]):
            return "api"
        if any(k in t for k in ["students", "lti"]):
            return "education"
        return "conversation_management"

    elif company == "Visa":
        if any(k in t for k in ["fraud", "identity", "stolen"]):
            return "fraud_support"
        if any(k in t for k in ["charge", "refund", "merchant", "dispute"]):
            return "dispute"
        if any(k in t for k in ["travel", "blocked", "cash"]):
            return "travel_support"
        return "card_support"

    return "general"


def should_escalate(text: str, min_length: int = MIN_TICKET_LENGTH_FOR_ESCALATION) -> bool:
    """
    Determine if the ticket requires immediate human escalation due to
    critical outage / security alerts or ambiguous/underspecified descriptions.
    """
    t = text.lower()

    if any(k in t for k in [
        "site is down",
        "security vulnerability",
        "all requests are failing",
        "stopped working completely"
    ]):
        return True

    if len(t.strip()) < min_length:
        return True

    return False
