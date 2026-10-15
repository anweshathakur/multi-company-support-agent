import sys
from pathlib import Path

# Ensure 'code' and root directories are present in sys.path
CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
for p in (str(CODE_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from pipeline import process_tickets
except ImportError:
    from code.pipeline import process_tickets


def main():
    print("=== STARTING SUPPORT TRIAGE AGENT ===")
    process_tickets()


if __name__ == "__main__":
    main()