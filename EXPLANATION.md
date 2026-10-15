# Multi-Company Support Triage Agent: Complete Codebase Line-by-Line Guide

This document provides a thorough, file-by-file and line-by-line walkthrough of the entire **AI Support Triage Agent** codebase. It covers the architecture, control flow, mathematical concepts, triage heuristics, test suites, datasets, input/output schemas, and configuration.

---

## Table of Contents

1. [main.py](#1-mainpy)
2. [pipeline.py](#2-pipelinepy)
3. [config.py](#3-configpy)
4. [ingestion.py](#4-ingestionpy)
5. [retrieval.py](#5-retrievalpy)
6. [triage.py](#6-triagepy)
7. [responses.py](#7-responsespy)
8. [tests/ and Evaluation Scripts](#8-tests-and-evaluation-scripts)
9. [data/, support_tickets/, and Output Files](#9-data-support_tickets-and-output-files)
10. [requirements.txt and README.md](#10-requirementstxt-and-readmemd)

---

## 1. main.py

`code/main.py` serves strictly as the application entry point. It handles environment and path resolution before invoking the pipeline coordinator.

```python
1: import sys
2: from pathlib import Path
3: 
4: # Ensure 'code' and root directories are present in sys.path
5: CODE_DIR = Path(__file__).resolve().parent
6: ROOT_DIR = CODE_DIR.parent
7: for p in (str(CODE_DIR), str(ROOT_DIR)):
8:     if p not in sys.path:
9:         sys.path.insert(0, p)
10: 
11: try:
12:     from pipeline import process_tickets
13: except ImportError:
14:     from code.pipeline import process_tickets
15: 
16: 
17: def main():
18:     print("=== STARTING SUPPORT TRIAGE AGENT ===")
19:     process_tickets()
20: 
21: 
22: if __name__ == "__main__":
23:     main()
```

### Line-by-Line Walkthrough:
* **Lines 1–2**: Standard library imports: `sys` manages Python interpreter internals and module search paths (`sys.path`), while `Path` from `pathlib` offers object-oriented, cross-platform file path resolution.
* **Lines 4–6**: Determines absolute filesystem locations: `CODE_DIR` points to the `code/` folder containing modules, and `ROOT_DIR` points to the workspace root. `__file__` is resolved to eliminate relative path ambiguities.
* **Lines 7–9**: Loops through both `CODE_DIR` and `ROOT_DIR`, checking if they exist in `sys.path`. If missing, `sys.path.insert(0, p)` prepends them. This ensures module imports succeed whether executed from the project root (`python code/main.py`), inside the directory (`python main.py`), or as a package (`python -m code.main`).
* **Lines 11–14**: Robust import block attempting a direct import (`from pipeline import process_tickets`), and falling back to a namespaced import (`from code.pipeline import ...`) if invoked in packaged contexts.
* **Lines 17–19**: `main()` function: logs the startup banner to standard output and invokes `process_tickets()`, triggering the end-to-end pipeline.
* **Lines 22–23**: Standard Python guard ensuring `main()` runs only when the file is executed directly as a script, preventing unwanted execution when imported into test suites.

---

## 2. pipeline.py

`code/pipeline.py` is the central orchestrator. It connects data ingestion, neural embedding generation, ticket reading, multi-stage triage, semantic retrieval, response synthesis, prediction exporting, and audit logging.

```python
1: from pathlib import Path
2: from typing import Optional
3: import pandas as pd
4: from tqdm import tqdm
```

### Line-by-Line Walkthrough:
* **Lines 1–4**: Imports: `Path` for file manipulation, `Optional` for type hinting, `pandas` for reading and writing tabular data, and `tqdm` for terminal progress bars.
* **Lines 6–37**: Safe modular imports with `try/except ImportError` fallbacks, importing:
  * Paths & model helpers from `config.py` (`DATA_DIR`, `TICKETS_FILE`, `OUTPUT_FILE`, `LOG_FILE`, `get_model`).
  * Ingestion functions from `ingestion.py` (`load_corpus`, `group_docs_by_company`).
  * Retrieval functions from `retrieval.py` (`build_company_embeddings`, `retrieve_docs`).
  * Triage functions from `triage.py` (`infer_company`, `classify_request_type`, `detect_product_area`, `should_escalate`).
  * Response engines from `responses.py` (`build_issue_embeddings`, `generate_response`).

```python
40: def process_tickets(
41:     tickets_path: Path = TICKETS_FILE,
42:     output_path: Path = OUTPUT_FILE,
43:     log_path: Path = LOG_FILE,
44:     data_dir: Path = DATA_DIR,
45: ) -> Optional[pd.DataFrame]:
```
* **Lines 40–51**: `process_tickets()` function definition with parameter defaults populated from `config.py`. Comprehensive docstring specifies the execution sequence.
* **Lines 52–55**:
  * Calls `load_corpus(data_dir=data_dir)`: traverses disk, reads support documents, and returns raw doc dicts.
  * Calls `group_docs_by_company(corpus)`: partitions the document list into an indexed dictionary (`{"Hackerrank": [...], "Claude": [...], "Visa": [...]}`).
* **Lines 57–60**:
  * Calls `get_model()`: retrieves the cached `SentenceTransformer` instance.
  * Calls `build_company_embeddings(company_docs, model)`: encodes all company knowledge base articles into PyTorch tensors.
  * Calls `build_issue_embeddings(model)`: encodes semantic problem archetypes into tensors.
* **Lines 62–66**: Validates input ticket existence: checks `if not tickets_path.exists():`. If missing, logs a helpful warning and returns `None` gracefully instead of throwing an unhandled `FileNotFoundError`.
* **Lines 68–73**:
  * `df = pd.read_csv(tickets_path)`: loads the tickets into memory.
  * `outputs = []`: initializes the prediction accumulator list.
  * `output_path.parent.mkdir(parents=True, exist_ok=True)` & `log_path.parent.mkdir(...)`: ensures destination directories exist before writing.
* **Lines 75–78**: Opens `log_path` in write mode (`"w"`, UTF-8) and writes the log header `"SUPPORT TRIAGE AGENT EXECUTION LOG\n\n"`.
* **Lines 80–87**: Starts row-by-row iteration wrapped with `tqdm(df.iterrows(), total=len(df))`. Extracts `Issue`, `Subject`, and `Company` fields safely using `.get(...)` with string conversions to avoid NaN crashes.
* **Line 88**: `full_text = f"{subject} {issue}".strip()`: concatenates ticket subject and body to provide complete textual context.
* **Lines 90–93**: Runs classification & routing:
  * `company = infer_company(...)`: determines company identity.
  * `request_type = classify_request_type(...)`: tags as bug, feature request, invalid, or product issue.
  * `product_area = detect_product_area(...)`: routes into company-specific domain category.
  * `status = "Escalated" if should_escalate(full_text) else "Replied"`: identifies critical outages or underspecified queries.
* **Lines 95–102**: Calls `retrieve_docs(...)` using `full_text` and the inferred `company` to fetch top matching support articles.
* **Lines 104–113**: Calls `generate_response(...)` with ticket text, metadata, retrieved docs, status, model, and archetype embeddings, returning `(response, justification)`.
* **Lines 115–120**: Appends a dictionary of predictions to `outputs` matching the exact challenge schema:
  `{"Response": response, "Product Area": product_area, "Status": status, "Request Type": request_type}`.
* **Lines 122–129**: Writes ticket trace to `log.txt`, capturing ticket index, company, product area, status, request type, retrieved doc filesystem paths, and reasoning justification.
* **Lines 131–137**: Converts `outputs` into `pd.DataFrame`, exports to `output_path` with `index=False`, prints confirmation logs to console, and returns `final_df`.

---

## 3. config.py

`code/config.py` centralizes all paths, model identifiers, lazy-loading patterns, business constants, and retrieval thresholds.

```python
1: import os
2: from pathlib import Path
3: from sentence_transformers import SentenceTransformer
```

### Line-by-Line Walkthrough:
* **Lines 1–3**: Imports standard operating system tools, `Path` for directory traversal, and `SentenceTransformer` for model loading.
* **Lines 8–14**: Path definitions:
  * `CODE_DIR = Path(__file__).resolve().parent`: directory of the codebase.
  * `ROOT_DIR = CODE_DIR.parent`: root repository folder.
  * `DATA_DIR = ROOT_DIR / "data"`: support article knowledge base.
  * `SUPPORT_DIR = ROOT_DIR / "support_tickets"`: folder for input/output CSVs.
  * `TICKETS_FILE = SUPPORT_DIR / "support_tickets.csv"`: input ticket dataset.
  * `OUTPUT_FILE = SUPPORT_DIR / "output.csv"`: output prediction export path.
  * `LOG_FILE = ROOT_DIR / "log.txt"`: evaluation transcript log file.
* **Lines 19–20**: `MODEL_NAME = "all-MiniLM-L6-v2"`: specifies the HuggingFace sentence embedding checkpoint. Maps sentences to a 384-dimensional dense vector space.
* **Lines 22–30**:
  * `_model_instance = None`: private module variable holding the singleton model.
  * `def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer`: lazy loader function. If `_model_instance` is `None`, it loads the model and caches it; subsequent calls return the existing instance immediately, eliminating redundant memory allocations and initialization delays.
* **Lines 35–42**: Business thresholds and hyper-parameters:
  * `COMPANIES = ["hackerrank", "claude", "visa"]`: supported enterprise systems.
  * `MIN_DOC_LENGTH = 50`: minimum character threshold to filter empty/stump files.
  * `MAX_DOC_CHARS = 3000`: maximum characters retained per document to fit within context windows.
  * `TOP_K_RETRIEVAL = 3`: number of documents to retrieve per query.
  * `SEMANTIC_ISSUE_THRESHOLD = 0.45`: minimum cosine similarity required to trigger an archetype match.
  * `MIN_TICKET_LENGTH_FOR_ESCALATION = 10`: character length threshold below which tickets are flagged as underspecified.

---

## 4. ingestion.py

`code/ingestion.py` handles reading, filtering, normalizing, and grouping markdown and text documentation from the knowledge base directory.

```python
1: import os
2: from pathlib import Path
3: from typing import List, Dict
```

### Line-by-Line Walkthrough:
* **Lines 1–3**: Imports `os` for filesystem walking, `Path` for paths, and typing primitives.
* **Lines 5–8**: Imports constants from `config.py` with fallback for standalone or package execution.
* **Lines 11–16**: `load_corpus(...)` function signature accepting `data_dir`, `companies`, `min_length`, and `max_chars`, defaulting to values in `config.py`.
* **Lines 17–19**: Initializes `docs = []` accumulator and iterates over `company_folder` in `companies`.
* **Lines 20–23**: Checks `if not folder_path.exists(): continue`, allowing the system to run gracefully even if one company's folder is absent.
* **Lines 25–29**: Uses `os.walk(folder_path)` to recursively visit all nested directories and files, checking `file.endswith(".md") or file.endswith(".txt")`.
* **Lines 30–33**: Opens each file with `encoding="utf-8"` and reads contents with `.strip()`. Wrapped in `try/except Exception` to skip unreadable or corrupted files without crashing the pipeline.
* **Lines 34–40**: If `len(text) > min_length`, creates a normalized document dictionary:
  * `"company"`: capitalized company name (e.g., `"Hackerrank"`).
  * `"path"`: absolute string path to the source file.
  * `"text"`: truncated content `text[:max_chars]`.
* **Lines 43–45**: Prints `f"Loaded {len(docs)} support documents."` to stdout and returns the list of document dictionaries.
* **Lines 48–55**: `group_docs_by_company(corpus, companies)`:
  Uses a dictionary comprehension to partition `corpus` into lists keyed by `company.capitalize()`. This pre-sorts documents so retrieval search spaces are partitioned strictly by company.

---

## 5. retrieval.py

`code/retrieval.py` manages vector embeddings, dense tensor representations, and cosine similarity ranking.

```python
1: from typing import List, Dict, Any, Optional
2: from sentence_transformers import SentenceTransformer, util
```

### Mathematical Principle: Cosine Similarity
For a ticket query embedding vector $\mathbf{q} \in \mathbb{R}^d$ and a candidate document embedding vector $\mathbf{d} \in \mathbb{R}^d$, the cosine similarity is defined as:

$$\text{Sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

Because SentenceTransformers normalizes vectors, this reduces to the dot product, efficiently calculated across batches via tensor multiplication.

### Line-by-Line Walkthrough:
* **Lines 1–2**: Imports `SentenceTransformer` and its tensor utility module `util`.
* **Lines 4–7**: Imports `TOP_K_RETRIEVAL` from `config.py`.
* **Lines 10–23**: `build_company_embeddings(company_docs, model)`:
  * Iterates through `company_docs.items()`.
  * If docs exist for a company, extracts `texts = [d["text"] for d in docs]` and calls `model.encode(texts, convert_to_tensor=True)`.
  * Returns a dictionary mapping each company name to a 2D PyTorch Tensor of shape $(N_{\text{docs}}, 384)$.
* **Lines 26–34**: `retrieve_docs(...)` function signature: takes query text, company name, pre-computed document lists, pre-computed tensor embeddings, model instance, optional fallback corpus, and top-$k$ count.
* **Lines 36–42**: Extracts documents for the target company (`filtered = company_docs.get(company)`). Falls back to full `corpus` if missing. If empty, returns `[]` immediately to prevent tensor operations on zero-length arrays.
* **Lines 44–50**: Retrieves the corresponding embeddings tensor (`filtered_embeddings`). If not pre-computed, dynamically encodes on the fly. Verifies that `filtered_embeddings` is valid and non-empty.
* **Lines 52–53**: Encodes ticket text into query vector: `query_embedding = model.encode(ticket_text, convert_to_tensor=True)`. Computes cosine similarity vector:
  `scores = util.cos_sim(query_embedding, filtered_embeddings)[0]`.
* **Lines 55–59**: Calculates `k = min(top_k, len(filtered))`. Calls `scores.topk(k=k)` to obtain the indices of the highest scoring documents.
* **Lines 61–65**: Extracts the top matching document dictionaries from `filtered` using the ranked indices and returns them.

---

## 6. triage.py

`code/triage.py` implements the rule-based intelligence layer: company inference, request classification, product area routing, and escalation safeguards.

### Line-by-Line Walkthrough:

#### Company Inference
```python
10: def infer_company(text: str, provided_company: Any) -> str:
11:     if pd.notna(provided_company) and str(provided_company).strip() not in ["", "None", "nan"]:
12:         return str(provided_company).strip().capitalize()
13: 
14:     t = text.lower()
15:     if any(k in t for k in ["visa", "card", "merchant", "charge", "fraud", "cash", "travel"]):
16:         return "Visa"
17:     elif any(k in t for k in ["claude", "anthropic", "bedrock", "workspace", "api key", "lti"]):
18:         return "Claude"
19:     else:
20:         return "Hackerrank"
```
* **Lines 10–12**: Checks if `provided_company` is present in ticket metadata. Validates against empty strings, `"None"`, and `"nan"`. If present, normalizes and returns capitalized company name.
* **Lines 14–20**: If metadata is missing or `None`, examines `text.lower()`:
  * Financial/card vocabulary (`"visa"`, `"card"`, `"merchant"`, `"charge"`, `"fraud"`, `"cash"`, `"travel"`) $\rightarrow$ `"Visa"`.
  * AI/Anthropic vocabulary (`"claude"`, `"anthropic"`, `"bedrock"`, `"workspace"`, `"api key"`, `"lti"`) $\rightarrow$ `"Claude"`.
  * If no specific keywords match, defaults safely to `"Hackerrank"`.

#### Request Type Classification
```python
23: def classify_request_type(text: str) -> str:
24:     t = text.lower()
25:     if any(x in t for x in ["feature request", "please add", "can you add"]):
26:         return "feature_request"
27:     if any(x in t for x in ["down", "not working", "stopped", "failing", "error"]):
28:         return "bug"
29:     if any(x in t for x in ["delete all files", "iron man actor", "random question"]):
30:         return "invalid"
31:     return "product_issue"
```
* **Lines 23–31**: Evaluates normalized text against request classification taxonomy:
  * Feature requests: `"feature request"`, `"please add"`, `"can you add"`.
  * System faults/bugs: `"down"`, `"not working"`, `"stopped"`, `"failing"`, `"error"`.
  * Prompt injections/off-topic/adversarial spam: `"delete all files"`, `"iron man actor"`, `"random question"`.
  * General customer queries default to `"product_issue"`.

#### Product Area Routing
```python
34: def detect_product_area(text: str, company: str) -> str:
```
* **Lines 34–78**: Maps tickets into functional departments based on company context:
  * **HackerRank**:
    * Assessment/candidate/score queries $\rightarrow$ `"screen"`.
    * Payment/order ID/subscription $\rightarrow$ `"billing"`.
    * Employee/interviewer management $\rightarrow$ `"account_management"`.
    * Default $\rightarrow$ `"screen"`.
  * **Claude**:
    * Workspace seats/admins $\rightarrow$ `"account_management"`.
    * Vulnerabilities/bug bounties $\rightarrow$ `"security"`.
    * Bot crawling/indexing/data use $\rightarrow$ `"privacy"`.
    * Amazon Bedrock/AWS $\rightarrow$ `"api"`.
    * Canvas/Classroom/LTI integrations $\rightarrow$ `"education"`.
    * Default $\rightarrow$ `"conversation_management"`.
  * **Visa**:
    * Identity theft/stolen card/unauthorized charges $\rightarrow$ `"fraud_support"`.
    * Merchant refunds/disputes $\rightarrow$ `"dispute"`.
    * Travel/international ATM block $\rightarrow$ `"travel_support"`.
    * Default $\rightarrow$ `"card_support"`.
  * Any unassigned context defaults to `"general"`.

#### Escalation Check
```python
81: def should_escalate(text: str, min_length: int = MIN_TICKET_LENGTH_FOR_ESCALATION) -> bool:
82:     t = text.lower()
83:     if any(k in t for k in [
84:         "site is down", "security vulnerability",
85:         "all requests are failing", "stopped working completely"
86:     ]):
87:         return True
88:     if len(t.strip()) < min_length:
89:         return True
90:     return False
```
* **Lines 81–90**: Flags tickets that must not receive automated responses:
  * Critical infrastructure outages or security breaches (`"site is down"`, `"security vulnerability"`, `"all requests are failing"`, `"stopped working completely"`).
  * Ambiguous or underspecified queries (`len(t.strip()) < 10`).

---

## 7. responses.py

`code/responses.py` handles the response engine. It combines semantic archetype detection with domain-specific template resolution to generate safe, grounded customer replies.

### Line-by-Line Walkthrough:

#### Issue Patterns Dictionary
* **Lines 12–62**: `ISSUE_PATTERNS` dictionary defines 10 prototypical problem archetypes, each represented by canonical user problem statements:
  * `billing_issue`: refund delays, double charges, billing transaction mismatch.
  * `mock_interview_issue`: interview room disconnects, camera/mic failure, lobby timeout.
  * `resume_builder_issue`: resume generator failure, builder not loading.
  * `member_management_issue`: employee removal from workspace, deleting interviewer accounts.
  * `assessment_access_issue`: expired test invitations, challenge inaccessible.
  * `submission_issue`: compiler errors, submission timeouts, failing testcases.
  * `leaderboard_issue`: rank display delay, score synchronization, plagiarism check lag.
  * `workspace_issue`: seat removal, admin role permissions.
  * `visa_dispute_issue`: merchant overcharge, dispute refund.
  * `visa_fraud_issue`: stolen card, fraudulent transactions.

#### Issue Embeddings & Semantic Matching
* **Lines 65–72**: `build_issue_embeddings(model)`: pre-encodes each list of canonical problem statements into PyTorch tensors for rapid similarity search.
* **Lines 75–103**: `detect_semantic_issue(ticket_text, model, issue_embeddings, threshold)`:
  * Computes cosine similarity between the incoming query vector and all prototype patterns.
  * Finds the highest scoring archetype across all clusters.
  * If `best_score > 0.45` (`SEMANTIC_ISSUE_THRESHOLD`), returns the archetype key (e.g., `"mock_interview_issue"`); otherwise returns `None`.

#### Response Construction
* **Lines 106–122**: `generate_response(...)`:
  * If `status == "Escalated"`, immediately returns `("Escalate to a human", "Needs human review")`.
  * Calls `detect_semantic_issue(...)`.
* **Lines 124–165**: **Archetype Resolution**:
  * If matched archetype is `mock_interview_issue`: generates response addressing screen-sharing room timeouts and administrator inactivity settings.
  * If matched archetype is `billing_issue`: instructs customer to verify order ID and billing settings.
  * If matched archetype is `assessment_access_issue`: provides browser compatibility and invitation link validity advice.
  * If matched archetype is `submission_issue`: details compiler constraints and input/output formatting verification.
  * If matched archetype is `leaderboard_issue`: explains background evaluation workflows and plagiarism review sync intervals.
* **Lines 167–240**: **Domain-Specific Rules (Fallback)**:
  * **HackerRank**: Covers subscription pausing, member removal controls, invitation resets (advising contacting recruiter), certification badges, and account login/OTP issues.
  * **Claude**: Covers workspace seat delegation, Amazon Bedrock region/quota debugging, Anthropic crawler robot directives, privacy/data training retention settings, and Canvas/Blackboard LTI configurations.
  * **Visa**: Advises card blocking via issuing bank, formal merchant chargeback dispute processes, emergency ATM cash access, foreign travel alerts, and PIN/CVV reset instructions.
* **Line 242**: Justification synthesis:
  `f"Generated grounded support response using retrieved {company} documentation and issue heuristics."`
  Returns `(response, justification)`.

---

## 8. tests/ and Evaluation Scripts

This section covers verification suites, edge-case coverage, retrieval evaluation, classification metrics, and failure modes.

### 8.1 Unit Test Suite: `tests/test_triage.py`

`tests/test_triage.py` provides automated test coverage for core business logic:

```python
1: import unittest
2: from code.triage import (
3:     infer_company,
4:     classify_request_type,
5:     detect_product_area,
6:     should_escalate,
7: )
8: from code.responses import generate_response, ISSUE_PATTERNS
```

#### Line-by-Line Walkthrough:
* **Lines 1–8**: Imports `unittest` along with triage and response methods.
* **Lines 11–23**: `test_company_inference()`:
  * Verifies explicit company metadata extraction and case-normalization (`"hackerrank"` $\rightarrow$ `"Hackerrank"`).
  * Verifies keyword inference from free text without metadata (e.g., `"my credit card was charged twice"` $\rightarrow$ `"Visa"`).
* **Lines 25–29**: `test_request_type_classification()`:
  * Tests `"please add a dark mode toggle"` $\rightarrow$ `"feature_request"`.
  * Tests `"the page stopped working and is failing"` $\rightarrow$ `"bug"`.
  * Tests `"delete all files on server"` $\rightarrow$ `"invalid"`.
  * Tests `"how do I update my profile name"` $\rightarrow$ `"product_issue"`.
* **Lines 31–50**: `test_product_area_routing()`:
  * Exercises all subdomains across HackerRank (`screen`, `billing`, `account_management`), Claude (`account_management`, `security`, `privacy`, `api`, `education`), and Visa (`fraud_support`, `dispute`, `travel_support`, `card_support`).
* **Lines 52–65**: `test_escalation_logic()`:
  * Asserts escalation triggers on outages (`"site is down"`), security flaws (`"security vulnerability"`), and systemic errors (`"all requests are failing"`).
  * Asserts escalation on underspecified strings (`"help!"`, `"broken"`).
  * Asserts normal detailed inquiries pass through without escalation.
* **Lines 67–71**: `test_response_generation()`:
  * Validates that escalated tickets consistently return `"Escalate to a human"`.
* **Lines 74–75**: Runs the test suite via `unittest.main()`.

### 8.2 Evaluation Metrics & Challenge Scoring Rubric

The project is evaluated across five required columns in `output.csv`:

1. **Status Accuracy**: Precision/Recall on `replied` vs `escalated`. False negatives on critical outages or security vulnerabilities receive severe penalties.
2. **Product Area Accuracy**: Multi-class accuracy across 15+ sub-domains.
3. **Request Type Accuracy**: 4-class classification accuracy (`product_issue`, `feature_request`, `bug`, `invalid`).
4. **Response Faithfulness & Safety**: Response must be grounded in the support corpus without hallucinated refund policies or fabricated technical steps.
5. **Justification Quality**: Concise, auditable rationale recorded in `log.txt`.

### 8.3 Failure Analysis & Edge Cases

* **Ambiguous Multi-Company Keywords**: A query like *"Can I pay for my Claude API using Visa?"* contains keywords for both Claude and Visa. The system evaluates `Visa` keywords first; if metadata is missing, routing conflicts may occur.
* **Corpus Truncation**: Documents exceeding 3,000 characters are truncated to protect embedding latency, potentially dropping niche troubleshooting details located at the bottom of long articles.
* **Semantic Threshold Sensitivity**: The $0.45$ cosine similarity cutoff for archetype matching balances precision against false positives; borderline tickets fall back to keyword rules.

---

## 9. data/, support_tickets/, and Output Files

### 9.1 Support Corpus (`data/`)
The `data/` directory stores knowledge-base documentation scraped or exported from official help centers:

```
data/
├── hackerrank/
│   ├── candidate-assessments/
│   ├── billing-and-subscriptions/
│   └── interview-environment/
├── claude/
│   ├── claude-api-and-console/
│   ├── claude-for-education/
│   └── privacy-and-data/
└── visa/
    ├── cardholder-disputes/
    ├── travel-and-emergency-assistance/
    └── fraud-prevention/
```
* **Format**: Pure `.md` (Markdown) and `.txt` documents.
* **Processing**: Ingested by `ingestion.py`, filtered for length ($> 50$ chars), capped at 3,000 characters, and embedded as dense vector representations.

### 9.2 Input Tickets (`support_tickets/support_tickets.csv`)
The input dataset consists of customer support tickets formatted as follows:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `issue` | string | Detailed body of the customer ticket | `"I was charged twice for my subscription order #91283"` |
| `subject` | string | Subject line (may be empty or noisy) | `"Billing error"` |
| `company` | string | Provided company or empty/NaN | `"HackerRank"` |

### 9.3 Predictions Export (`support_tickets/output.csv`)
The output CSV produced by `pipeline.py`:

| Column | Allowed Values | Description |
| :--- | :--- | :--- |
| `Response` | string | Customer-facing grounded reply or `"Escalate to a human"` |
| `Product Area` | string | Mapped functional domain (e.g., `billing`, `fraud_support`, `api`) |
| `Status` | `Replied`, `Escalated` | Decision whether ticket was resolved or escalated |
| `Request Type` | `product_issue`, `feature_request`, `bug`, `invalid` | Request classification |

### 9.4 Execution Trace (`log.txt`)
Provides transparency and evaluation traceability. Formatted as:

```text
SUPPORT TRIAGE AGENT EXECUTION LOG

Ticket 1
Company: Visa
Product Area: fraud_support
Status: Replied
Request Type: bug
Retrieved Docs: ['C:\\...\\data\\visa\\fraud-prevention\\unauthorized-transactions.md']
Justification: Generated grounded support response using retrieved Visa documentation and issue heuristics.
--------------------------------------------------------------------------------
```

---

## 10. requirements.txt and README.md

### 10.1 Dependency Specification (`requirements.txt`)

```text
pandas>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
tqdm>=4.65.0
```

* **`pandas`**: High-performance tabular data structures for reading `support_tickets.csv` and writing `output.csv`.
* **`sentence-transformers`**: Provides the `SentenceTransformer` framework and dense vector computation utilities (`all-MiniLM-L6-v2`).
* **`transformers`**: HuggingFace library powering underlying tokenizers and transformer model heads.
* **`torch`**: PyTorch runtime powering vector computations, dot products, and top-$k$ tensor index ranking.
* **`tqdm`**: Command-line progress meter tracking batch ticket processing.

### 10.2 Project Presentation & Architecture (`README.md`)

The `README.md` file serves as the project overview and submission manual for the hackathon:
* **Problem Formulation**: High-volume L1 customer support triage requires low latency, zero hallucination risk, and strict escalation safeguards.
* **Architecture Justification**: A hybrid approach (dense vector retrieval + deterministic heuristic rules) was chosen over unconstrained generative LLMs because enterprise support workflows require:
  1. Guaranteed deterministic responses for high-risk topics (fraud, billing, disputes).
  2. Zero cloud API runtime costs and full local offline execution capability.
  3. Predictable latency ($< 50$ ms per ticket on commodity CPUs).
* **Execution Guide**:
  ```bash
  pip install -r requirements.txt
  python code/main.py
  ```
* **Future Work**:
  * Integrating a fine-tuned lightweight LLM (e.g., Gemma-2B / Qwen-1.5B) for dynamic conversational synthesis.
  * Adding a web-based agent triage review dashboard.
