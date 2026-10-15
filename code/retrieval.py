from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util

try:
    from config import TOP_K_RETRIEVAL
except ImportError:
    from code.config import TOP_K_RETRIEVAL


def build_company_embeddings(
    company_docs: Dict[str, List[Dict[str, str]]],
    model: SentenceTransformer
) -> Dict[str, Any]:
    """
    Pre-compute dense tensor embeddings for each company's support documentation.
    """
    company_embeddings = {}
    for company, docs in company_docs.items():
        if docs:
            texts = [d["text"] for d in docs]
            company_embeddings[company] = model.encode(texts, convert_to_tensor=True)
        else:
            company_embeddings[company] = None
    return company_embeddings


def retrieve_docs(
    ticket_text: str,
    company: str,
    company_docs: Dict[str, List[Dict[str, str]]],
    company_embeddings: Dict[str, Any],
    model: SentenceTransformer,
    corpus: Optional[List[Dict[str, str]]] = None,
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, str]]:
    """
    Retrieve top-k most relevant documents for a ticket query using cosine similarity
    over pre-computed company embeddings.
    """
    filtered = company_docs.get(company)
    if filtered is None and corpus is not None:
        filtered = corpus
    elif filtered is None:
        filtered = []

    if not filtered:
        return []

    if company in company_embeddings and company_embeddings[company] is not None:
        filtered_embeddings = company_embeddings[company]
    else:
        filtered_embeddings = model.encode([d["text"] for d in filtered], convert_to_tensor=True)

    if filtered_embeddings is None or len(filtered_embeddings) == 0:
        return []

    query_embedding = model.encode(ticket_text, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, filtered_embeddings)[0]

    k = min(top_k, len(filtered))
    if k == 0:
        return []

    top_results = scores.topk(k=k)

    retrieved = []
    for idx in top_results.indices:
        retrieved.append(filtered[int(idx)])

    return retrieved
