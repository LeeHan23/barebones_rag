"""
Vector store — ChromaDB (file-based, no server) replacing pgvector.
Carries over TopicBoostedRetriever + TOPIC_HINTS from bare_NutriChatbot.
"""
import os
from typing import List, Set

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from embeddings import get_embedding_function

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/ext/barebones_rag/data/chroma")

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC TAXONOMY
# Maps query phrases → topic tags stored in chunk metadata.
# On a cache hit, chunks tagged with the matched topics get a score boost.
# Add entries freely — both English and Bahasa Malaysia phrases are supported.
# ─────────────────────────────────────────────────────────────────────────────
TOPIC_HINTS = {
    "hypertension":            {"hypertension", "blood pressure", "hypertension management"},
    "high blood pressure":     {"hypertension", "blood pressure"},
    "blood pressure":          {"hypertension", "blood pressure"},
    "darah tinggi":            {"hypertension", "blood pressure"},
    "heart failure":           {"heart failure", "HF management"},
    "fluid restriction":       {"heart failure", "HF management", "CKD"},
    "cholesterol":             {"cholesterol", "lipid management", "dyslipidaemia"},
    "lipid":                   {"lipid management", "dyslipidaemia"},
    "ldl":                     {"cholesterol", "lipid management"},
    "trans fat":               {"cholesterol management", "fats"},
    "saturated fat":           {"cholesterol management", "fats"},
    " fat ":                   {"cholesterol management", "fats"},
    "lemak":                   {"cholesterol management", "fats"},
    "sodium":                  {"sodium intake", "sodium reduction", "DASH diet"},
    "salt":                    {"sodium intake", "sodium reduction"},
    "garam":                   {"sodium intake", "sodium reduction"},
    "diabetes":                {"diabetes", "T2DM", "diabetes management", "blood sugar"},
    "blood sugar":             {"blood sugar", "diabetes management"},
    "kencing manis":           {"diabetes", "T2DM"},
    "gula":                    {"diabetes", "blood sugar", "sugar intake"},
    "coronary":                {"CAD", "coronary artery disease"},
    "heart attack":            {"CAD", "coronary heart disease", "post-MI care"},
    "angina":                  {"CAD", "stable angina"},
    "exercise":                {"physical activity", "exercise", "cardiac rehabilitation"},
    "physical activity":       {"physical activity", "exercise"},
    "senaman":                 {"physical activity", "exercise"},
    "smoking":                 {"smoking cessation", "tobacco"},
    "merokok":                 {"smoking cessation", "tobacco"},
    "depression":              {"depression", "mental health"},
    "stress":                  {"psychological health", "stress"},
    "diet":                    {"nutrition", "diet", "heart-healthy diet"},
    "nutrition":               {"nutrition", "diet"},
    "makan":                   {"nutrition", "diet"},
    "food":                    {"nutrition", "diet"},
    "sugar":                   {"sugar intake", "diabetes management"},
    "cvd":                     {"CVD prevention", "cardiovascular health"},
    "cardiovascular":          {"CVD prevention", "cardiovascular health"},
    "heart disease":           {"CVD prevention", "coronary heart disease"},
    "chronic kidney":          {"CKD", "renal nutrition"},
    "kidney disease":          {"CKD", "renal nutrition"},
    " ckd ":                   {"CKD", "renal nutrition"},
    "potassium":               {"CKD", "potassium restriction"},
    "phosphorus":              {"CKD", "phosphorus restriction"},
    "obesity":                 {"obesity", "weight management"},
    "overweight":              {"obesity", "weight management"},
    "berat badan":             {"obesity", "weight management"},
    "sleep":                   {"sleep", "healthy lifestyle"},
    "pcos":                    {"PCOS", "insulin resistance"},
    "insulin resistance":      {"insulin resistance", "T2DM"},
}


def detect_query_topics(query: str) -> Set[str]:
    """Return the union of all topic tags matched by the query string."""
    q = " " + query.lower() + " "
    matched: Set[str] = set()
    for phrase, topics in TOPIC_HINTS.items():
        if phrase in q:
            matched.update(topics)
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVERS
# ─────────────────────────────────────────────────────────────────────────────
class TopicBoostedRetriever(BaseRetriever):
    """
    Wraps a base retriever and re-ranks results by topic overlap.

    1. Fetch a wide candidate pool from the base retriever
    2. Detect query topics from TOPIC_HINTS + patient_conditions
    3. Score: base_rank_score + boost_factor * overlap_ratio
    4. Return top_k

    Chunks without topic metadata keep their embedding rank — the boost is
    additive, not exclusive.
    """
    base_retriever: BaseRetriever
    top_k: int = 5
    boost_factor: float = 0.5
    patient_conditions: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        query_topics = detect_query_topics(query)
        for cond in self.patient_conditions:
            query_topics |= detect_query_topics(cond)

        if not query_topics:
            return candidates[: self.top_k]

        scored = []
        for rank, doc in enumerate(candidates):
            base_score = 1.0 / (rank + 1)
            doc_topics = set(doc.metadata.get("doc_topics", []))
            overlap = len(query_topics & doc_topics) / len(query_topics) if query_topics else 0.0
            scored.append((base_score + self.boost_factor * overlap, rank, doc))

        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[: self.top_k]
        n_boosted = sum(
            1 for _, _, d in top
            if set(d.metadata.get("doc_topics", [])) & query_topics
        )
        print(
            f"[TopicBoost] topics={sorted(query_topics)} | "
            f"boosted {n_boosted}/{self.top_k}"
        )
        return [doc for _, _, doc in top]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FACTORY
# ─────────────────────────────────────────────────────────────────────────────
_chroma_db = None


def _get_chroma() -> Chroma:
    global _chroma_db
    if _chroma_db is None:
        _chroma_db = Chroma(
            collection_name="knowledge",
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )
    return _chroma_db


def get_retriever(patient_conditions: List[str] = None) -> BaseRetriever:
    """
    Returns a TopicBoostedRetriever over the single ChromaDB knowledge collection.
    patient_conditions adds extra topic signals for reranking (e.g. ["Hypertension", "T2DM"]).
    """
    db = _get_chroma()
    base_retriever = db.as_retriever(search_kwargs={"k": 15})
    return TopicBoostedRetriever(
        base_retriever=base_retriever,
        top_k=5,
        boost_factor=0.5,
        patient_conditions=patient_conditions or [],
    )


def get_chroma_for_ingest() -> Chroma:
    """Returns the Chroma instance for direct document ingestion."""
    return _get_chroma()
