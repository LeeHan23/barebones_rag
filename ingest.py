#!/usr/bin/env python3
"""
CLI bulk ingestion — feed a file or directory into the ChromaDB knowledge store.

Usage:
    python ingest.py /path/to/docs/
    python ingest.py /path/to/file.pdf
    python ingest.py /path/to/docs/ --chunk-size 600 --overlap 80
"""
import argparse
import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ingestion import SUPPORTED_EXTENSIONS as SUPPORTED
from ingestion import chunk_text, extract_text
from vector_store import get_chroma_for_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-ingest documents into the RAG store")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    files = (
        [f for f in target.rglob("*") if f.suffix.lower() in SUPPORTED]
        if target.is_dir()
        else [target]
    )
    if not files:
        print("No supported files found (.pdf .docx .txt .md)")
        sys.exit(0)

    db = get_chroma_for_ingest()
    total_added = 0

    for f in sorted(files):
        if f.suffix.lower() not in SUPPORTED:
            continue
        try:
            raw = f.read_bytes()
            text = extract_text(f.suffix.lower(), raw)
            chunks = chunk_text(text, args.chunk_size, args.overlap)
            if not chunks:
                print(f"  skip  {f.name} — no text extracted")
                continue

            file_hash = hashlib.sha256(raw).hexdigest()[:16]
            ids = [f"{file_hash}_{i}" for i in range(len(chunks))]

            existing = set(db._collection.get(ids=ids)["ids"])
            new = [(c, i) for c, i in zip(chunks, ids) if i not in existing]

            if new:
                texts, new_ids = zip(*new)
                db.add_texts(list(texts), ids=list(new_ids))

            skipped = len(chunks) - len(new)
            print(f"  ✓  {f.name}: +{len(new)} chunks ({skipped} already exist)")
            total_added += len(new)
        except Exception as e:
            print(f"  ✗  {f.name}: {e}")

    print(f"\nDone — {total_added} new chunks added to store at {db._persist_directory}")


if __name__ == "__main__":
    main()
