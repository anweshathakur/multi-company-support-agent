import os
from pathlib import Path
from typing import List, Dict

try:
    from config import DATA_DIR, COMPANIES, MIN_DOC_LENGTH, MAX_DOC_CHARS
except ImportError:
    from code.config import DATA_DIR, COMPANIES, MIN_DOC_LENGTH, MAX_DOC_CHARS


def load_corpus(
    data_dir: Path = DATA_DIR,
    companies: List[str] = COMPANIES,
    min_length: int = MIN_DOC_LENGTH,
    max_chars: int = MAX_DOC_CHARS,
) -> List[Dict[str, str]]:
    """
    Traverse the data directory for each company, read markdown and text files,
    and return normalized support document records.
    """
    docs = []

    for company_folder in companies:
        folder_path = data_dir / company_folder

        if not folder_path.exists():
            continue

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()

                            if len(text) > min_length:
                                docs.append({
                                    "company": company_folder.capitalize(),
                                    "path": str(file_path),
                                    "text": text[:max_chars]
                                })
                    except Exception:
                        continue

    print(f"Loaded {len(docs)} support documents.")
    return docs


def group_docs_by_company(
    corpus: List[Dict[str, str]],
    companies: List[str] = COMPANIES,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Group the loaded corpus documents by capitalized company name.
    """
    return {
        company.capitalize(): [d for d in corpus if d["company"] == company.capitalize()]
        for company in companies
    }
