"""
Shared document-ingestion helpers — text extraction and chunking.
Used by both the /ingest API endpoint (app.py) and the bulk CLI (ingest.py)
so the two paths can never drift apart.
"""
import io

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def extract_text(ext: str, content: bytes) -> str:
    """Pull plain text out of a supported file type given its extension and raw bytes."""
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    return content.decode("utf-8", errors="replace")


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping word-count chunks.

    `step` is clamped to at least 1 — if overlap >= size the naive
    `size - overlap` step would never advance `start`, looping forever.
    """
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        c = " ".join(words[start : start + size])
        if c.strip():
            chunks.append(c)
    return chunks
