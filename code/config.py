import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
SUPPORT_DIR = ROOT_DIR / "support_tickets"
TICKETS_FILE = SUPPORT_DIR / "support_tickets.csv"
OUTPUT_FILE = SUPPORT_DIR / "output.csv"
LOG_FILE = ROOT_DIR / "log.txt"

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

MODEL_NAME = "all-MiniLM-L6-v2"

_model_instance = None


def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Lazy-loads and caches the SentenceTransformer model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(model_name)
    return _model_instance


# ==============================================================================
# BUSINESS & TRIAGE THRESHOLDS
# ==============================================================================

COMPANIES = ["hackerrank", "claude", "visa"]

# Retrieval parameters
MIN_DOC_LENGTH = 50
MAX_DOC_CHARS = 3000
TOP_K_RETRIEVAL = 3

# Triage & Classification thresholds
SEMANTIC_ISSUE_THRESHOLD = 0.45
MIN_TICKET_LENGTH_FOR_ESCALATION = 10
