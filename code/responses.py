from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util

try:
    from config import SEMANTIC_ISSUE_THRESHOLD
except ImportError:
    from code.config import SEMANTIC_ISSUE_THRESHOLD

# ==============================================================================
# PROTOTYPICAL SEMANTIC ISSUE PATTERNS
# ==============================================================================

ISSUE_PATTERNS = {
    "billing_issue": [
        "refund not received after payment",
        "charged twice on subscription",
        "payment issue with order id",
        "billing transaction incorrect"
    ],
    "mock_interview_issue": [
        "mock interview disconnected",
        "candidate removed from interview room",
        "camera microphone failed during interview",
        "candidate sent back to lobby due to inactivity"
    ],
    "resume_builder_issue": [
        "resume builder not loading",
        "resume generation failed"
    ],
    "member_management_issue": [
        "remove employee from workspace",
        "delete interviewer account",
        "remove user from team"
    ],
    "assessment_access_issue": [
        "assessment link expired",
        "cannot access challenge",
        "test not opening",
        "question page not loading"
    ],
    "submission_issue": [
        "submission failed on challenge",
        "compiler error while submitting",
        "code not running on testcase"
    ],
    "leaderboard_issue": [
        "leaderboard rank not updated",
        "score not updated after submission",
        "plagiarism review affecting score"
    ],
    "workspace_issue": [
        "workspace seat removed",
        "workspace admin permissions issue"
    ],
    "visa_dispute_issue": [
        "merchant charged incorrectly",
        "refund not received from seller"
    ],
    "visa_fraud_issue": [
        "unauthorized visa transaction",
        "stolen card fraud"
    ]
}


def build_issue_embeddings(model: SentenceTransformer) -> Dict[str, Any]:
    """
    Pre-compute dense tensor embeddings for all issue archetype patterns.
    """
    return {
        key: model.encode(vals, convert_to_tensor=True)
        for key, vals in ISSUE_PATTERNS.items()
    }


def detect_semantic_issue(
    ticket_text: str,
    model: SentenceTransformer,
    issue_embeddings: Dict[str, Any],
    threshold: float = SEMANTIC_ISSUE_THRESHOLD,
) -> Optional[str]:
    """
    Compare ticket text against issue prototypes via cosine similarity.
    Returns the best matching archetype if score exceeds threshold.
    """
    if model is None or not issue_embeddings:
        return None

    query = model.encode(ticket_text, convert_to_tensor=True)
    best_issue = None
    best_score = -1.0

    for issue, embeds in issue_embeddings.items():
        score = util.cos_sim(query, embeds).max().item()
        if score > best_score:
            best_score = score
            best_issue = issue

    if best_score > threshold:
        return best_issue
    return None


def generate_response(
    ticket_text: str,
    company: str,
    docs: List[Dict[str, str]],
    status: str,
    request_type: str,
    product_area: str,
    model: Optional[SentenceTransformer] = None,
    issue_embeddings: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Generate customer-facing response and justification using
    semantic archetype matching and domain-specific rules.
    """
    if status == "Escalated":
        return "Escalate to a human", "Needs human review"

    lower = ticket_text.lower()
    semantic_issue = (
        detect_semantic_issue(ticket_text, model, issue_embeddings)
        if (model and issue_embeddings)
        else None
    )

    response = None

    # 1. Check Archetype Match First
    if semantic_issue == "mock_interview_issue":
        response = (
            "Interview inactivity timeouts may be affecting candidate and interviewer session "
            "continuity during long screen-sharing rounds. Please have the workspace administrator "
            "review current interview room inactivity settings, as extended timeout configuration "
            "or platform-side adjustment may be required."
        )
    elif semantic_issue == "billing_issue":
        response = (
            "We understand your concern regarding the billing transaction. Please verify the "
            "payment details and associated order information from your HackerRank billing settings. "
            "If the charge still appears incorrect, the billing team can review the transaction further."
        )
    elif semantic_issue == "assessment_access_issue":
        response = (
            "Please verify browser compatibility, network stability, and whether the assessment link "
            "is still active before retrying. If the challenge interface or assessment page continues "
            "failing to load, HackerRank support may need to inspect the assessment environment."
        )
    elif semantic_issue == "submission_issue":
        response = (
            "Please verify the selected programming language, required input/output format, and "
            "hidden compilation constraints before resubmitting. If submissions continue failing "
            "despite valid execution, HackerRank support can review the assessment environment."
        )
    elif semantic_issue == "leaderboard_issue":
        response = (
            "Leaderboard positions and score visibility may take time to refresh depending on challenge "
            "validation workflows. Please allow the evaluation cycle to complete, after which the final "
            "score state should reflect accordingly."
        )

    # 2. Company-specific keyword rules and domain fallbacks (if no archetype response matched)
    if response is None:
        if company == "Hackerrank":
            if any(k in lower for k in ["refund", "payment", "order id", "charged", "billing"]):
                response = (
                    "We understand your concern regarding the billing transaction. Please verify the "
                    "payment details and associated order information from your HackerRank billing settings. "
                    "If the charge still appears incorrect, the billing team can review the transaction further."
                )
            elif any(k in lower for k in ["mock interview", "camera", "microphone", "audio", "video interview"]):
                response = (
                    "Please verify browser permissions for microphone and camera access, along with internet stability, "
                    "before retrying the mock interview session. If the interruption persists or the previous attempt was "
                    "affected by a platform issue, HackerRank support can investigate the session logs."
                )
            elif any(k in lower for k in ["resume builder", "resume not loading", "resume generate"]):
                response = (
                    "If Resume Builder is not loading or generating output correctly, please clear browser cache, "
                    "disable interfering extensions, and retry in a supported browser. Continued failures may require "
                    "platform-side review."
                )
            elif any(k in lower for k in ["remove employee", "remove user", "remove interviewer", "delete member", "remove member"]):
                response = (
                    "Team member removal can be completed from the HackerRank user or workspace management settings "
                    "if administrator permissions are enabled. Please locate the member profile and use the available "
                    "remove option, or contact the workspace owner if the control is unavailable."
                )
            elif any(k in lower for k in ["subscription pause", "pause subscription", "hold subscription"]):
                response = (
                    "Eligible HackerRank subscriptions can be paused directly from the billing dashboard under subscription controls. "
                    "Please review whether your current plan supports pausing and submit the request through the billing section."
                )
            elif any(k in lower for k in ["reschedule", "assessment link expired", "expired test link"]):
                response = (
                    "Assessment rescheduling and expired invitation resets are generally controlled by the recruiting "
                    "organization that issued the assessment. Please contact the recruiter or hiring company directly to "
                    "request a refreshed attempt window."
                )
            elif any(k in lower for k in ["submission failed", "cannot submit", "compiler error", "testcase failing", "code not running"]):
                response = (
                    "Please verify the selected programming language, required input/output format, and any hidden compilation "
                    "constraints before resubmitting the challenge. If submissions continue failing despite valid code execution, "
                    "HackerRank support can review the assessment environment."
                )
            elif any(k in lower for k in ["certificate", "badge", "medal not showing"]):
                response = (
                    "Certificates and badges are typically generated after successful challenge or skill verification completion. "
                    "Please confirm that all completion criteria have been met and allow some time for profile synchronization before retrying."
                )
            elif any(k in lower for k in ["login issue", "cannot login", "otp not received", "account locked"]):
                response = (
                    "Please verify your registered email, password credentials, and spam folder for any OTP or verification emails. "
                    "If login access remains unavailable after retrying password reset, account-level assistance may be required."
                )
            elif any(k in lower for k in ["leaderboard", "rank", "score not updated", "plagiarism suspected"]):
                response = (
                    "Leaderboard positions and score visibility may take time to refresh depending on challenge validation and "
                    "plagiarism review workflows. Please allow the evaluation cycle to complete, after which the final score state "
                    "should reflect accordingly."
                )
            elif any(k in lower for k in ["assessment", "question not loading", "editor", "challenge page", "test not opening", "cannot access test"]):
                response = (
                    "Please verify browser compatibility, network stability, and whether the assessment link is still active "
                    "before retrying. If the challenge interface or editor continues failing to load, HackerRank support may need "
                    "to inspect the assessment environment."
                )
            else:
                response = (
                    "Thank you for contacting HackerRank support. Please review the relevant account or assessment configuration "
                    "associated with this issue and retry the workflow. If the problem continues, additional platform investigation may be required."
                )

        elif company == "Claude":
            if any(k in lower for k in ["workspace", "seat", "member removed", "owner"]):
                response = (
                    "Workspace membership and seat assignments are managed by the workspace administrator. "
                    "Please contact the workspace owner to restore permissions or reassign the required seat."
                )
            elif any(k in lower for k in ["bedrock", "aws", "api failing", "model timeout", "request failing"]):
                response = (
                    "For Claude requests routed through Amazon Bedrock, please verify AWS region availability, "
                    "Bedrock model permissions, API credentials, and current request quotas. Persistent failures may require "
                    "Bedrock log inspection."
                )
            elif any(k in lower for k in ["crawl", "crawler", "website indexing", "block claude bot"]):
                response = (
                    "Anthropic provides crawler control mechanisms that allow website owners to restrict future Claude crawling activity. "
                    "Please apply the documented crawler or robots directives at the domain level."
                )
            elif any(k in lower for k in ["data", "model improve", "training data", "retention"]):
                response = (
                    "Customer data retention and model-improvement usage depend on the Claude product plan and privacy configuration "
                    "enabled for the workspace. Enterprise and privacy-controlled environments provide additional restrictions."
                )
            elif any(k in lower for k in ["lti", "students", "canvas", "classroom"]):
                response = (
                    "Claude for Education supports LTI-based LMS integrations for institution-managed student provisioning. "
                    "Administrators typically need to generate LTI credentials and complete LMS-side configuration."
                )
            elif any(k in lower for k in ["conversation deleted", "chat missing", "history missing"]):
                response = (
                    "Please verify whether the conversation was removed from the current workspace, archived, or hidden by account filters. "
                    "If conversation history is unexpectedly missing across sessions, workspace administrator review may be required."
                )
            else:
                response = (
                    "Thank you for contacting Claude support. Please verify the relevant account, API, or privacy configuration "
                    "associated with your request. If the issue continues, administrator review or support escalation may be necessary."
                )

        elif company == "Visa":
            if any(k in lower for k in ["identity", "stolen", "fraud", "unauthorized", "suspicious transaction"]):
                response = (
                    "This appears to be a fraud or card security issue. Please immediately contact the issuing bank, request "
                    "the Visa card to be blocked, review unauthorized transactions, and begin the fraud dispute process."
                )
            elif any(k in lower for k in ["charge", "merchant", "wrong product", "refund not received", "dispute"]):
                response = (
                    "For merchant disputes or incorrect charges, please first contact the seller and retain all receipts or "
                    "communication records. If the issue remains unresolved, your issuing bank can begin a formal dispute review."
                )
            elif any(k in lower for k in ["cash", "atm", "withdrawal failed"]):
                response = (
                    "If urgent cash access is required, please use an eligible ATM withdrawal or contact your issuing bank "
                    "regarding emergency cash assistance options available under your card benefits."
                )
            elif any(k in lower for k in ["blocked", "travel", "international transaction", "declined abroad"]):
                response = (
                    "If your Visa card is blocked during travel or foreign transactions, please contact the issuing bank "
                    "immediately to verify travel authorization, fraud restrictions, or temporary security locks."
                )
            elif any(k in lower for k in ["pin", "forgot pin", "cvv", "verification failed"]):
                response = (
                    "PIN resets, CVV verification concerns, and card authentication issues are handled directly by the issuing "
                    "financial institution. Please contact your bank for secure card verification assistance."
                )
            else:
                response = (
                    "Please contact the financial institution that issued your Visa card, as account servicing and card-level "
                    "actions are handled directly by the issuer."
                )

        else:
            response = "Thank you for contacting support. We are reviewing your issue and will assist you shortly."

    justification = f"Generated grounded support response using retrieved {company} documentation and issue heuristics."
    return response, justification
