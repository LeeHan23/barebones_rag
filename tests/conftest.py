"""
Shared fixtures for API tests.

The real app loads a ~2GB embedding model and a ChromaDB collection on first
use, and `/chat` calls out to a live LLM. None of that is needed to verify
routing, validation, and response shaping — so these fixtures swap in fakes
at the points app.py touches the outside world.
"""
import sys
import types

import pytest


class FakeCollection:
    def __init__(self):
        self._docs: dict[str, str] = {}

    def get(self, ids=None):
        if ids is None:
            return {"ids": list(self._docs)}
        return {"ids": [i for i in ids if i in self._docs]}

    def count(self):
        return len(self._docs)

    def delete(self, ids):
        for i in ids:
            self._docs.pop(i, None)


class FakeChroma:
    """Stand-in for langchain_community.vectorstores.Chroma used by app.py."""

    def __init__(self):
        self._collection = FakeCollection()
        self._persist_directory = "/fake/chroma/path"

    def add_texts(self, texts, ids):
        for text, doc_id in zip(texts, ids):
            self._collection._docs[doc_id] = text


@pytest.fixture
def fake_chroma():
    return FakeChroma()


@pytest.fixture
def client(monkeypatch, fake_chroma):
    """TestClient for app.py with the vector store and RAG chain mocked out."""
    # `app` imports `rag_stream` and `get_chroma_for_ingest` by name, so patch
    # the references inside the app module — patching the source module would
    # be too late, app already holds its own reference.
    import app as app_module

    monkeypatch.setattr(app_module, "get_chroma_for_ingest", lambda: fake_chroma)
    monkeypatch.setattr(
        app_module,
        "rag_stream",
        lambda question, session_id: iter([f"echo: {question}"]),
    )

    from fastapi.testclient import TestClient

    return TestClient(app_module.app)
