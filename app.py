import hashlib
import io
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import rag_stream
from vector_store import get_chroma_for_ingest

app = FastAPI(title="Barebones RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ── Models ─────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str
    session_id: str = ""


class ChatResponse(BaseModel):
    answer: str
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    try:
        db = get_chroma_for_ingest()
        count = db._collection.count()
    except Exception:
        count = -1
    return {"status": "ok", "chunks_in_store": count}


@app.post("/chat")
def chat(body: ChatRequest):
    """Streaming SSE chat. Each chunk is `data: <text>\\n\\n`, ending with `data: [DONE]\\n\\n`."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")

    session_id = body.session_id or str(uuid.uuid4())

    def generate():
        for chunk in rag_stream(body.question, session_id):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id},
    )


@app.post("/chat/sync", response_model=ChatResponse)
def chat_sync(body: ChatRequest):
    """Non-streaming version — returns the full answer as JSON."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")

    session_id = body.session_id or str(uuid.uuid4())
    answer = "".join(rag_stream(body.question, session_id))
    return ChatResponse(answer=answer, session_id=session_id)


@app.post("/ingest")
async def ingest(file: UploadFile):
    """Upload a PDF, DOCX, TXT, or MD file and add it to the knowledge store."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {ext}")

    content = await file.read()
    text = _extract_text(ext, content)
    chunks = _chunk(text)

    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from this file")

    file_hash = hashlib.sha256(content).hexdigest()[:16]
    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]

    db = get_chroma_for_ingest()
    existing = set(db._collection.get(ids=ids)["ids"])
    new_chunks = [(c, i) for c, i in zip(chunks, ids) if i not in existing]

    if new_chunks:
        texts, new_ids = zip(*new_chunks)
        db.add_texts(list(texts), ids=list(new_ids))

    return {
        "filename": file.filename,
        "chunks_added": len(new_chunks),
        "chunks_skipped": len(chunks) - len(new_chunks),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_text(ext: str, content: bytes) -> str:
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    return content.decode("utf-8", errors="replace")


def _chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap
    return chunks
