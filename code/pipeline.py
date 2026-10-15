from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

try:
    from config import (
        DATA_DIR,
        TICKETS_FILE,
        OUTPUT_FILE,
        LOG_FILE,
        get_model,
    )
    from ingestion import load_corpus, group_docs_by_company
    from retrieval import build_company_embeddings, retrieve_docs
    from triage import (
        infer_company,
        classify_request_type,
        detect_product_area,
        should_escalate,
    )
    from responses import build_issue_embeddings, generate_response
except ImportError:
    from code.config import (
        DATA_DIR,
        TICKETS_FILE,
        OUTPUT_FILE,
        LOG_FILE,
        get_model,
    )
    from code.ingestion import load_corpus, group_docs_by_company
    from code.retrieval import build_company_embeddings, retrieve_docs
    from code.triage import (
        infer_company,
        classify_request_type,
        detect_product_area,
        should_escalate,
    )
    from code.responses import build_issue_embeddings, generate_response


def process_tickets(
    tickets_path: Path = TICKETS_FILE,
    output_path: Path = OUTPUT_FILE,
    log_path: Path = LOG_FILE,
    data_dir: Path = DATA_DIR,
) -> Optional[pd.DataFrame]:
    """
    Coordinate the end-to-end triage pipeline:
    1. Load support documentation corpus and group by company
    2. Build semantic embeddings for company docs and issue archetypes
    3. Read input tickets CSV
    4. For each ticket: infer company, classify type, detect product area,
       evaluate escalation, retrieve grounded docs, and generate response
    5. Save predictions to output CSV and trace logs to log file
    """
    print("Loading support corpus and initializing embeddings...")
    corpus = load_corpus(data_dir=data_dir)
    company_docs = group_docs_by_company(corpus)

    model = get_model()
    company_embeddings = build_company_embeddings(company_docs, model)
    issue_embeddings = build_issue_embeddings(model)

    if not tickets_path.exists():
        print(f"Warning: Ticket file not found at {tickets_path}.")
        print("Please place the target CSV in the support_tickets/ folder.")
        return None

    print(f"Reading support tickets from {tickets_path}...")
    df = pd.read_csv(tickets_path)
    outputs = []

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("SUPPORT TRIAGE AGENT EXECUTION LOG\n\n")

        print("Beginning ticket processing...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            print(f"Processing ticket {idx+1}")

            issue = str(row.get("Issue", ""))
            subject = str(row.get("Subject", ""))
            provided_company = row.get("Company", "")

            full_text = f"{subject} {issue}".strip()

            company = infer_company(full_text, provided_company)
            request_type = classify_request_type(full_text)
            product_area = detect_product_area(full_text, company)
            status = "Escalated" if should_escalate(full_text) else "Replied"
            docs = retrieve_docs(
                ticket_text=full_text,
                company=company,
                company_docs=company_docs,
                company_embeddings=company_embeddings,
                model=model,
                corpus=corpus,
            )

            response, justification = generate_response(
                ticket_text=full_text,
                company=company,
                docs=docs,
                status=status,
                request_type=request_type,
                product_area=product_area,
                model=model,
                issue_embeddings=issue_embeddings,
            )

            outputs.append({
                "Response": response,
                "Product Area": product_area,
                "Status": status,
                "Request Type": request_type,
            })

            log.write(f"Ticket {idx+1}\n")
            log.write(f"Company: {company}\n")
            log.write(f"Product Area: {product_area}\n")
            log.write(f"Status: {status}\n")
            log.write(f"Request Type: {request_type}\n")
            log.write(f"Retrieved Docs: {[d['path'] for d in docs]}\n")
            log.write(f"Justification: {justification}\n")
            log.write("-" * 80 + "\n")

    final_df = pd.DataFrame(outputs)
    final_df.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")
    print(f"Saved logs to {log_path}")

    return final_df
