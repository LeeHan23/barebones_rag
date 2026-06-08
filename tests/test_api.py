"""
API integration tests for app.py.

The vector store and the RAG chain are mocked (see conftest.py) so these run
fast and don't require an embedding model, ChromaDB collection, or a live LLM.
"""
import io


# ── /health ────────────────────────────────────────────────────────────────


def test_health_reports_chunk_count(client, fake_chroma):
    fake_chroma.add_texts(["a", "b"], ids=["id1", "id2"])

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "chunks_in_store": 2}


def test_health_returns_minus_one_when_store_unavailable(monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    def broken_store():
        raise RuntimeError("store offline")

    monkeypatch.setattr(app_module, "get_chroma_for_ingest", broken_store)
    resp = TestClient(app_module.app).get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "chunks_in_store": -1}


# ── /chat ──────────────────────────────────────────────────────────────────


def test_chat_rejects_empty_question(client):
    resp = client.post("/chat", json={"question": "   "})
    assert resp.status_code == 400


def test_chat_streams_sse_chunks_and_done_marker(client):
    resp = client.post("/chat", json={"question": "hello", "session_id": "s1"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-session-id"] == "s1"
    body = resp.text
    assert "data: echo: hello" in body
    assert body.strip().endswith("data: [DONE]")


def test_chat_generates_session_id_when_absent(client):
    resp = client.post("/chat", json={"question": "hello"})
    assert resp.headers["x-session-id"]  # non-empty UUID string


# ── /chat/sync ─────────────────────────────────────────────────────────────


def test_chat_sync_returns_full_answer(client):
    resp = client.post("/chat/sync", json={"question": "hello", "session_id": "s2"})

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"answer": "echo: hello", "session_id": "s2"}


def test_chat_sync_rejects_empty_question(client):
    resp = client.post("/chat/sync", json={"question": ""})
    assert resp.status_code == 400


# ── /ingest ────────────────────────────────────────────────────────────────


def test_ingest_rejects_unsupported_extension(client):
    resp = client.post(
        "/ingest",
        files={"file": ("notes.exe", io.BytesIO(b"binary"), "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_ingest_rejects_file_with_no_extractable_text(client):
    resp = client.post(
        "/ingest",
        files={"file": ("empty.txt", io.BytesIO(b"   \n\n  "), "text/plain")},
    )
    assert resp.status_code == 422


def test_ingest_adds_new_chunks(client, fake_chroma):
    content = b"Hypertension management requires reducing sodium intake daily."
    resp = client.post(
        "/ingest",
        files={"file": ("guide.txt", io.BytesIO(content), "text/plain")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "guide.txt"
    assert body["chunks_added"] == 1
    assert body["chunks_skipped"] == 0
    assert fake_chroma._collection.count() == 1


def test_ingest_skips_already_known_chunks(client, fake_chroma):
    content = b"Hypertension management requires reducing sodium intake daily."
    files = {"file": ("guide.txt", io.BytesIO(content), "text/plain")}

    first = client.post("/ingest", files=files)
    second = client.post(
        "/ingest",
        files={"file": ("guide.txt", io.BytesIO(content), "text/plain")},
    )

    assert first.json()["chunks_added"] == 1
    assert second.json()["chunks_added"] == 0
    assert second.json()["chunks_skipped"] == 1
    assert fake_chroma._collection.count() == 1
