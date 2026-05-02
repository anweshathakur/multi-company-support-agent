# AI Support Triage Agent

An intelligent retrieval-based customer support automation system built for the HackerRank May 26 Orchestrate Hackathon.

This agent reads unresolved support tickets, identifies the correct company context, classifies the issue, retrieves the most relevant support documentation, and generates a grounded customer-facing support reply.

The project is designed to simulate how enterprise support teams can automate first-level ticket triage while still maintaining high-quality, human-like responses.

---

## Problem Statement

Large support teams spend significant manual effort on:

- identifying which product/company a ticket belongs to
- understanding the nature of the issue
- searching internal documentation
- drafting repetitive customer replies

This creates slower response times and higher operational load.

Our solution automates this first-response triage layer using semantic retrieval and intelligent response orchestration.

---

## Features

- Automatic company inference (HackerRank / Claude / Visa)
- Request type classification
- Product area detection
- Escalation decision logic
- Semantic retrieval over 700+ support documents
- Human-like grounded support response generation
- Execution logging for transparency
- CSV prediction export for evaluation

---

## Project Architecture

1. Load all support documentation from company knowledge bases.
2. Build semantic embeddings using SentenceTransformers.
3. Read incoming support tickets CSV.
4. Infer company if missing.
5. Classify request type.
6. Detect product area.
7. Decide whether human escalation is required.
8. Retrieve top relevant support documents.
9. Generate customer-facing response.
10. Export final structured output.

---

## Tech Stack

- Python
- Pandas
- SentenceTransformers
- HuggingFace Transformers
- Cosine Similarity Retrieval
- Markdown Knowledge Base Files

---

## Folder Structure

```bash
hackerrank-orchestrate-may26/

├── code/
│   └── main.py

├── data/
│   ├── hackerrank/
│   ├── claude/
│   └── visa/

├── support_tickets/
│   ├── support_tickets.csv
│   └── output.csv

├── log.txt
└── README.md
```

---

## How To Run

Install dependencies:

```bash
pip install pandas sentence-transformers transformers torch tqdm
```

Run the agent:

```bash
py code/main.py
```

The system generates:

- support_tickets/output.csv
- log.txt

---

## Output Columns

The generated CSV contains:

- Response
- Product Area
- Status
- Request Type

---

## Intelligent Decision Layers

### Company Inference

Uses keyword heuristics and provided metadata to determine the most likely company context.

### Semantic Retrieval

Uses dense vector similarity search to fetch the most relevant support documents from the internal corpus.

### Product Area Detection

Maps the ticket into product domains such as billing, dispute, account management, fraud support, privacy, and education.

### Escalation Layer

Critical outages, security vulnerabilities, or underspecified issues are automatically marked for human review.

### Human Response Generation

Generates concise customer-facing replies using retrieved support knowledge and issue-specific heuristics.

---

## Future Improvements

- Replace heuristic response layer with a fine-tuned lightweight LLM
- Add web dashboard for support agents
