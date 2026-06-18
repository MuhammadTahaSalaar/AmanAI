"""
Seed script: load data/processed/all_documents.json → embed → upsert into Supabase.

Usage:
    python -m ingestion.seed
    python -m ingestion.seed --path /custom/path/docs.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path when run as a module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.ingest import content_hash, ingest_items  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion.seed")

DEFAULT_PATH = ROOT / "data" / "processed" / "all_documents.json"


def load_documents(path: Path) -> list[dict]:
    logger.info("Loading documents from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at top level")
    logger.info("Loaded %d documents", len(data))
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed AmanAI document store")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to JSON file (default: data/processed/all_documents.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Embedding batch size (default: 20)",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        logger.error("File not found: %s", args.path)
        sys.exit(1)

    docs = load_documents(args.path)
    result = ingest_items(docs, source_kind="seed")
    logger.info(
        "Seeding complete — added: %d, skipped (duplicates): %d",
        result["added"],
        result["skipped"],
    )


if __name__ == "__main__":
    main()
