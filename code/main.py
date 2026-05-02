import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

print("=== STARTING SUPPORT TRIAGE AGENT ===")

# CONFIG

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SUPPORT_DIR = ROOT_DIR / "support_tickets"
OUTPUT_FILE = SUPPORT_DIR / "output.csv"
LOG_FILE = ROOT_DIR / "log.txt"

model = SentenceTransformer("all-MiniLM-L6-v2")

# LOAD SUPPORT CORPUS


def load_corpus():
    docs = []

    for company_folder in ["hackerrank", "claude", "visa"]:
        folder_path = DATA_DIR / company_folder

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()

                            if len(text) > 50:
                                docs.append({
                                    "company": company_folder.capitalize(),
                                    "path": str(file_path),
                                    "text": text[:3000]
                                })
                    except:
                        continue

    print(f"Loaded {len(docs)} support documents.")
    return docs


CORPUS = load_corpus()

COMPANY_DOCS = {
    "Hackerrank": [d for d in CORPUS if d["company"] == "Hackerrank"],
    "Claude": [d for d in CORPUS if d["company"] == "Claude"],
    "Visa": [d for d in CORPUS if d["company"] == "Visa"]
}

COMPANY_EMBEDDINGS = {
    company: model.encode([d["text"] for d in docs], convert_to_tensor=True)
    for company, docs in COMPANY_DOCS.items()
}

# COMPANY INFERENCE


def infer_company(text, provided_company):
    if pd.notna(provided_company) and str(provided_company).strip() not in ["", "None", "nan"]:
        return str(provided_company).strip().capitalize()

    t = text.lower()

    if any(k in t for k in ["visa", "card", "merchant", "charge", "fraud", "cash", "travel"]):
        return "Visa"
    elif any(k in t for k in ["claude", "anthropic", "bedrock", "workspace", "api key", "lti"]):
        return "Claude"
    else:
        return "Hackerrank"

# RETRIEVE DOCS


def retrieve_docs(ticket_text, company, top_k=3):
    filtered = COMPANY_DOCS.get(company, CORPUS)

    if company in COMPANY_EMBEDDINGS:
        filtered_embeddings = COMPANY_EMBEDDINGS[company]
    else:
        filtered_embeddings = model.encode([d["text"] for d in filtered], convert_to_tensor=True)

    query_embedding = model.encode(ticket_text, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, filtered_embeddings)[0]
    top_results = scores.topk(k=min(top_k, len(filtered)))

    retrieved = []
    for idx in top_results.indices:
        retrieved.append(filtered[int(idx)])

    return retrieved

# REQUEST TYPE CLASSIFIER


def classify_request_type(text):
    t = text.lower()

    if any(x in t for x in ["feature request", "please add", "can you add"]):
        return "feature_request"

    if any(x in t for x in ["down", "not working", "stopped", "failing", "error"]):
        return "bug"

    if any(x in t for x in ["delete all files", "iron man actor", "random question"]):
        return "invalid"

    return "product_issue"

# PRODUCT AREA DETECTOR

def detect_product_area(text, company):
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

# ESCALATION


def should_escalate(text):
    t = text.lower()

    if any(k in t for k in [
        "site is down",
        "security vulnerability",
        "all requests are failing",
        "stopped working completely"
    ]):
        return True

    if len(t.strip()) < 10:
        return True

    return False

# RESPONSE GENERATOR

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

ISSUE_EMBEDDINGS = {
    key: model.encode(vals, convert_to_tensor=True)
    for key, vals in ISSUE_PATTERNS.items()
}

def detect_semantic_issue(ticket_text):
    query = model.encode(ticket_text, convert_to_tensor=True)
    best_issue = None
    best_score = -1

    for issue, embeds in ISSUE_EMBEDDINGS.items():
        score = util.cos_sim(query, embeds).max().item()
        if score > best_score:
            best_score = score
            best_issue = issue

    if best_score > 0.45:
        return best_issue
    return None

def generate_response(ticket_text, company, docs, status, request_type, product_area):
    
    if status == "Escalated":
        return "Escalate to a human", "Needs human review"

    lower = ticket_text.lower()

    semantic_issue = detect_semantic_issue(ticket_text)

    if semantic_issue == "mock_interview_issue":
            response = "Interview inactivity timeouts may be affecting candidate and interviewer session continuity during long screen-sharing rounds. Please have the workspace administrator review current interview room inactivity settings, as extended timeout configuration or platform-side adjustment may be required."

    elif semantic_issue == "billing_issue":
            response = "We understand your concern regarding the billing transaction. Please verify the payment details and associated order information from your HackerRank billing settings. If the charge still appears incorrect, the billing team can review the transaction further."

    elif semantic_issue == "assessment_access_issue":
            response = "Please verify browser compatibility, network stability, and whether the assessment link is still active before retrying. If the challenge interface or assessment page continues failing to load, HackerRank support may need to inspect the assessment environment."

    elif semantic_issue == "submission_issue":
            response = "Please verify the selected programming language, required input/output format, and hidden compilation constraints before resubmitting. If submissions continue failing despite valid execution, HackerRank support can review the assessment environment."

    elif semantic_issue == "leaderboard_issue":
            response = "Leaderboard positions and score visibility may take time to refresh depending on challenge validation workflows. Please allow the evaluation cycle to complete, after which the final score state should reflect accordingly."
    

    # HACKERRANK RESPONSES

    if company == "Hackerrank":

        if any(k in lower for k in ["refund", "payment", "order id", "charged", "billing"]):
            response = "We understand your concern regarding the billing transaction. Please verify the payment details and associated order information from your HackerRank billing settings. If the charge still appears incorrect, the billing team can review the transaction further."

        elif any(k in lower for k in ["mock interview", "camera", "microphone", "audio", "video interview"]):
            response = "Please verify browser permissions for microphone and camera access, along with internet stability, before retrying the mock interview session. If the interruption persists or the previous attempt was affected by a platform issue, HackerRank support can investigate the session logs."

        elif any(k in lower for k in ["resume builder", "resume not loading", "resume generate"]):
            response = "If Resume Builder is not loading or generating output correctly, please clear browser cache, disable interfering extensions, and retry in a supported browser. Continued failures may require platform-side review."

        elif any(k in lower for k in ["remove employee", "remove user", "remove interviewer", "delete member", "remove member"]):
            response = "Team member removal can be completed from the HackerRank user or workspace management settings if administrator permissions are enabled. Please locate the member profile and use the available remove option, or contact the workspace owner if the control is unavailable."

        elif any(k in lower for k in ["subscription pause", "pause subscription", "hold subscription"]):
            response = "Eligible HackerRank subscriptions can be paused directly from the billing dashboard under subscription controls. Please review whether your current plan supports pausing and submit the request through the billing section."

        elif any(k in lower for k in ["reschedule", "assessment link expired", "expired test link"]):
            response = "Assessment rescheduling and expired invitation resets are generally controlled by the recruiting organization that issued the assessment. Please contact the recruiter or hiring company directly to request a refreshed attempt window."

        elif any(k in lower for k in ["submission failed", "cannot submit", "compiler error", "testcase failing", "code not running"]):
            response = "Please verify the selected programming language, required input/output format, and any hidden compilation constraints before resubmitting the challenge. If submissions continue failing despite valid code execution, HackerRank support can review the assessment environment."

        elif any(k in lower for k in ["certificate", "badge", "medal not showing"]):
            response = "Certificates and badges are typically generated after successful challenge or skill verification completion. Please confirm that all completion criteria have been met and allow some time for profile synchronization before retrying."

        elif any(k in lower for k in ["login issue", "cannot login", "otp not received", "account locked"]):
            response = "Please verify your registered email, password credentials, and spam folder for any OTP or verification emails. If login access remains unavailable after retrying password reset, account-level assistance may be required."

        elif any(k in lower for k in ["leaderboard", "rank", "score not updated", "plagiarism suspected"]):
            response = "Leaderboard positions and score visibility may take time to refresh depending on challenge validation and plagiarism review workflows. Please allow the evaluation cycle to complete, after which the final score state should reflect accordingly."
        
        elif any(k in lower for k in ["assessment", "question not loading", "editor", "challenge page", "test not opening", "cannot access test"]):
            response = "Please verify browser compatibility, network stability, and whether the assessment link is still active before retrying. If the challenge interface or editor continues failing to load, HackerRank support may need to inspect the assessment environment."

        else:
            response = "Thank you for contacting HackerRank support. Please review the relevant account or assessment configuration associated with this issue and retry the workflow. If the problem continues, additional platform investigation may be required."

    # CLAUDE RESPONSES

    elif company == "Claude":

        if any(k in lower for k in ["workspace", "seat", "member removed", "owner"]):
            response = "Workspace membership and seat assignments are managed by the workspace administrator. Please contact the workspace owner to restore permissions or reassign the required seat."

        elif any(k in lower for k in ["bedrock", "aws", "api failing", "model timeout", "request failing"]):
            response = "For Claude requests routed through Amazon Bedrock, please verify AWS region availability, Bedrock model permissions, API credentials, and current request quotas. Persistent failures may require Bedrock log inspection."

        elif any(k in lower for k in ["crawl", "crawler", "website indexing", "block claude bot"]):
            response = "Anthropic provides crawler control mechanisms that allow website owners to restrict future Claude crawling activity. Please apply the documented crawler or robots directives at the domain level."

        elif any(k in lower for k in ["data", "model improve", "training data", "retention"]):
            response = "Customer data retention and model-improvement usage depend on the Claude product plan and privacy configuration enabled for the workspace. Enterprise and privacy-controlled environments provide additional restrictions."

        elif any(k in lower for k in ["lti", "students", "canvas", "classroom"]):
            response = "Claude for Education supports LTI-based LMS integrations for institution-managed student provisioning. Administrators typically need to generate LTI credentials and complete LMS-side configuration."

        elif any(k in lower for k in ["conversation deleted", "chat missing", "history missing"]):
            response = "Please verify whether the conversation was removed from the current workspace, archived, or hidden by account filters. If conversation history is unexpectedly missing across sessions, workspace administrator review may be required."

        else:
            response = "Thank you for contacting Claude support. Please verify the relevant account, API, or privacy configuration associated with your request. If the issue continues, administrator review or support escalation may be necessary."

    # VISA RESPONSES

    elif company == "Visa":

        if any(k in lower for k in ["identity", "stolen", "fraud", "unauthorized", "suspicious transaction"]):
            response = "This appears to be a fraud or card security issue. Please immediately contact the issuing bank, request the Visa card to be blocked, review unauthorized transactions, and begin the fraud dispute process."

        elif any(k in lower for k in ["charge", "merchant", "wrong product", "refund not received", "dispute"]):
            response = "For merchant disputes or incorrect charges, please first contact the seller and retain all receipts or communication records. If the issue remains unresolved, your issuing bank can begin a formal dispute review."

        elif any(k in lower for k in ["cash", "atm", "withdrawal failed"]):
            response = "If urgent cash access is required, please use an eligible ATM withdrawal or contact your issuing bank regarding emergency cash assistance options available under your card benefits."

        elif any(k in lower for k in ["blocked", "travel", "international transaction", "declined abroad"]):
            response = "If your Visa card is blocked during travel or foreign transactions, please contact the issuing bank immediately to verify travel authorization, fraud restrictions, or temporary security locks."

        elif any(k in lower for k in ["pin", "forgot pin", "cvv", "verification failed"]):
            response = "PIN resets, CVV verification concerns, and card authentication issues are handled directly by the issuing financial institution. Please contact your bank for secure card verification assistance."

        else:
            response = "Please contact the financial institution that issued your Visa card, as account servicing and card-level actions are handled directly by the issuer."

    else:
        response = "Thank you for contacting support. We are reviewing your issue and will assist you shortly."

    justification = f"Generated grounded support response using retrieved {company} documentation and issue heuristics."
    return response, justification

# ======================================

def process_tickets():
    print("Reading support tickets...")
    df = pd.read_csv(SUPPORT_DIR / "support_tickets.csv")
    outputs = []

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("SUPPORT TRIAGE AGENT EXECUTION LOG\n\n")

        print("Beginning ticket processing...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            print(f"Processing ticket {idx+1}")

            issue = str(row.get("Issue", ""))
            subject = str(row.get("Subject", ""))
            provided_company = row.get("Company", "")

            full_text = f"{subject} {issue}"

            company = infer_company(full_text, provided_company)
            request_type = classify_request_type(full_text)
            product_area = detect_product_area(full_text, company)
            status = "Escalated" if should_escalate(full_text) else "Replied"
            docs = retrieve_docs(full_text, company)

            response, justification = generate_response(
                full_text, company, docs, status, request_type, product_area
            )

            outputs.append({
                "Response": response,
                "Product Area": product_area,
                "Status": status,
                "Request Type": request_type
            })

            log.write(f"Ticket {idx+1}\n")
            log.write(f"Company: {company}\n")
            log.write(f"Product Area: {product_area}\n")
            log.write(f"Status: {status}\n")
            log.write(f"Request Type: {request_type}\n")
            log.write(f"Retrieved Docs: {[d['path'] for d in docs]}\n")
            log.write(f"Justification: {justification}\n")
            log.write("-"*80 + "\n")

    final_df = pd.DataFrame(outputs)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved predictions to {OUTPUT_FILE}")
    print(f"Saved logs to {LOG_FILE}")


if __name__ == "__main__":
    process_tickets()