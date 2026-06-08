"""
Unit tests for the topic taxonomy and TopicBoostedRetriever re-ranking logic.

These exercise pure logic only — no embedding model or ChromaDB is loaded,
so a fake base retriever stands in for the real one.
"""
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from vector_store import TopicBoostedRetriever, detect_query_topics


# ── detect_query_topics ────────────────────────────────────────────────────


def test_detect_query_topics_matches_known_phrase():
    topics = detect_query_topics("What should I eat for high blood pressure?")
    assert "hypertension" in topics
    assert "blood pressure" in topics


def test_detect_query_topics_matches_bahasa_malaysia_phrase():
    topics = detect_query_topics("Saya ada masalah darah tinggi")
    assert "hypertension" in topics


def test_detect_query_topics_no_match_returns_empty_set():
    assert detect_query_topics("What time is it in Tokyo?") == set()


def test_detect_query_topics_unions_multiple_phrases():
    topics = detect_query_topics("How does diabetes affect cholesterol and blood pressure?")
    assert {"diabetes", "cholesterol", "hypertension"} <= topics


# ── TopicBoostedRetriever ──────────────────────────────────────────────────


class _FakeRetriever(BaseRetriever):
    docs: list

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query, *, run_manager):
        return self.docs


def _make_retriever(docs, top_k=5, boost_factor=0.5, patient_conditions=None):
    return TopicBoostedRetriever(
        base_retriever=_FakeRetriever(docs=docs),
        top_k=top_k,
        boost_factor=boost_factor,
        patient_conditions=patient_conditions or [],
    )


def _invoke(retriever, query):
    # _get_relevant_documents requires a run_manager; build one via the public API
    return retriever.invoke(query)


def test_topic_match_promotes_doc_above_a_higher_base_ranked_non_match():
    # NOTE: with boost_factor=0.5, a rank-0 doc with zero overlap scores 1.0,
    # which a boosted lower-ranked doc can at best *tie* (0.5 base + 0.5 boost)
    # — and ties favour the lower rank index. So the boost cannot dethrone an
    # unrelated #1 result; it re-orders candidates *below* it instead.
    docs = [
        Document(page_content="top base-ranked but unrelated", metadata={"doc_topics": []}),
        Document(page_content="second base-ranked, also unrelated", metadata={"doc_topics": []}),
        Document(page_content="third base-ranked but matches the query topic", metadata={"doc_topics": ["hypertension", "blood pressure"]}),
    ]
    retriever = _make_retriever(docs, top_k=3, boost_factor=0.5)

    results = _invoke(retriever, "How do I manage my high blood pressure?")
    names = [d.page_content for d in results]

    assert names.index("third base-ranked but matches the query topic") < names.index(
        "second base-ranked, also unrelated"
    )


def test_no_query_topics_preserves_base_order():
    docs = [
        Document(page_content="first", metadata={"doc_topics": []}),
        Document(page_content="second", metadata={"doc_topics": ["hypertension"]}),
        Document(page_content="third", metadata={"doc_topics": []}),
    ]
    retriever = _make_retriever(docs, top_k=3)

    results = _invoke(retriever, "What's a good recipe for chicken soup?")

    assert [d.page_content for d in results] == ["first", "second", "third"]


def test_top_k_limits_results():
    docs = [Document(page_content=f"doc{i}", metadata={"doc_topics": []}) for i in range(10)]
    retriever = _make_retriever(docs, top_k=4)

    results = _invoke(retriever, "diabetes management tips")

    assert len(results) == 4


def test_empty_candidates_returns_empty_list():
    retriever = _make_retriever([], top_k=5)
    assert _invoke(retriever, "anything") == []


def test_patient_conditions_contribute_to_topic_matching():
    docs = [
        Document(page_content="rank0 unrelated", metadata={"doc_topics": []}),
        Document(page_content="rank1 unrelated", metadata={"doc_topics": []}),
        Document(page_content="CKD renal nutrition guide", metadata={"doc_topics": ["CKD", "renal nutrition"]}),
    ]
    retriever = _make_retriever(docs, top_k=3, patient_conditions=["Chronic Kidney Disease"])

    # The query alone matches no TOPIC_HINTS phrase — only the patient
    # condition surfaces "CKD"/"renal nutrition", promoting that doc above
    # the equally-unrelated doc that started one rank ahead of it.
    assert detect_query_topics("What can I cook for dinner tonight?") == set()

    results = _invoke(retriever, "What can I cook for dinner tonight?")
    names = [d.page_content for d in results]

    assert names.index("CKD renal nutrition guide") < names.index("rank1 unrelated")
