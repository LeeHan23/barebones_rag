#!/usr/bin/env python3
"""
Retrieval evaluation harness — a lightweight regression check for the RAG
retrieval layer (embeddings + ChromaDB + TopicBoostedRetriever).

Unlike tests/test_vector_store.py (which exercises the re-ranking *logic* with
fake documents and no model), this script runs the *real* embedding model
against a small golden corpus in a throwaway ChromaDB collection, then checks
that golden queries retrieve passages tagged with the expected topics.

It exists to catch regressions when you:
  - swap the embedding model (EMBEDDING_MODEL / EMBEDDING_ADAPTER_PATH)
  - change chunking, top_k, or boost_factor
  - edit TOPIC_HINTS in vector_store.py

Usage:
    python eval_retrieval.py            # human-readable report
    python eval_retrieval.py --json     # machine-readable report (for CI)

Exit code is non-zero if the hit rate falls below --min-hit-rate (default 0.7).
"""
import argparse
import contextlib
import json
import shutil
import sys
import tempfile

from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma

from embeddings import get_embedding_function
from vector_store import TopicBoostedRetriever

# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN CORPUS
# Each passage is tagged with the same `doc_topics` taxonomy used in production
# (see TOPIC_HINTS in vector_store.py) so the boost behaves realistically.
# ─────────────────────────────────────────────────────────────────────────────
GOLDEN_CORPUS = [
    {
        "id": "htn-1",
        "text": (
            "The DASH diet (Dietary Approaches to Stop Hypertension) emphasises "
            "fruits, vegetables, whole grains, and low-fat dairy while limiting "
            "sodium to under 2300mg per day to help lower blood pressure."
        ),
        "doc_topics": ["hypertension", "blood pressure", "DASH diet", "sodium intake"],
    },
    {
        "id": "diabetes-1",
        "text": (
            "Type 2 diabetes management centres on monitoring blood sugar levels, "
            "spreading carbohydrate intake evenly across meals, and choosing "
            "low-glycaemic-index foods such as wholegrain rice and legumes."
        ),
        "doc_topics": ["diabetes", "T2DM", "blood sugar", "diabetes management"],
    },
    {
        "id": "ckd-1",
        "text": (
            "People with chronic kidney disease often need to restrict potassium "
            "and phosphorus intake — limiting bananas, dairy, nuts, and processed "
            "foods with phosphate additives — under the guidance of a renal dietitian."
        ),
        "doc_topics": ["CKD", "renal nutrition", "potassium restriction", "phosphorus restriction"],
    },
    {
        "id": "cholesterol-1",
        "text": (
            "Lowering LDL cholesterol involves reducing saturated and trans fats — "
            "found in fried foods, ghee, and santan — and replacing them with "
            "unsaturated fats from fish, nuts, and olive oil."
        ),
        "doc_topics": ["cholesterol", "lipid management", "dyslipidaemia", "fats"],
    },
    {
        "id": "weight-1",
        "text": (
            "Sustainable weight management focuses on portion awareness — using "
            "the suku-suku-separuh quarter-plate method common in Malaysian "
            "households — alongside regular physical activity like brisk walking."
        ),
        "doc_topics": ["obesity", "weight management", "physical activity"],
    },
    {
        "id": "smoking-1",
        "text": (
            "Smoking cessation significantly reduces cardiovascular risk within "
            "weeks; nicotine replacement therapy and behavioural support both "
            "improve quit rates compared to going cold turkey alone."
        ),
        "doc_topics": ["smoking cessation", "tobacco", "CVD prevention"],
    },
    {
        "id": "general-1",
        "text": (
            "A balanced plate generally includes a quarter of protein, a quarter "
            "of carbohydrates, and half vegetables — a simple visual heuristic "
            "that works across most cultural cuisines without needing to weigh food."
        ),
        "doc_topics": ["nutrition", "diet"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN QUERIES
# `expect_topics`: any one of these topics appearing on a top-k result counts
# as a hit for that query (an OR match — phrasing varies, taxonomy may overlap).
# `expect_doc_id`: optional — if set, that specific passage must appear in top-k.
# ─────────────────────────────────────────────────────────────────────────────
GOLDEN_QUERIES = [
    {
        "query": "What foods should I eat if I have high blood pressure?",
        "expect_topics": {"hypertension", "blood pressure", "DASH diet"},
        "expect_doc_id": "htn-1",
    },
    {
        "query": "Saya ada masalah darah tinggi, makanan apa yang sesuai?",
        "expect_topics": {"hypertension", "blood pressure"},
        "expect_doc_id": "htn-1",
    },
    {
        "query": "How do I manage my blood sugar with type 2 diabetes?",
        "expect_topics": {"diabetes", "T2DM", "blood sugar", "diabetes management"},
        "expect_doc_id": "diabetes-1",
    },
    {
        "query": "I have kidney disease — should I avoid bananas and nuts?",
        "expect_topics": {"CKD", "renal nutrition", "potassium restriction"},
        "expect_doc_id": "ckd-1",
    },
    {
        "query": "What's the best way to lower my LDL and cholesterol levels?",
        "expect_topics": {"cholesterol", "lipid management", "dyslipidaemia"},
        "expect_doc_id": "cholesterol-1",
    },
    {
        "query": "Any tips for losing weight using the Malaysian plate method?",
        "expect_topics": {"obesity", "weight management", "physical activity"},
        "expect_doc_id": "weight-1",
    },
    {
        "query": "What's the best way to quit smoking for my heart health?",
        "expect_topics": {"smoking cessation", "tobacco", "CVD prevention"},
        "expect_doc_id": "smoking-1",
    },
]


_ID_BY_TEXT = {p["text"]: p["id"] for p in GOLDEN_CORPUS}


def build_eval_retriever(persist_dir: str, top_k: int = 3, boost_factor: float = 0.5):
    db = Chroma(
        collection_name="eval_knowledge",
        persist_directory=persist_dir,
        embedding_function=get_embedding_function(),
    )
    db.add_texts(
        texts=[p["text"] for p in GOLDEN_CORPUS],
        metadatas=[{"doc_topics": p["doc_topics"]} for p in GOLDEN_CORPUS],
        ids=[p["id"] for p in GOLDEN_CORPUS],
    )
    # Wide candidate pool (the whole corpus) so the topic boost has room to
    # re-rank; `top_k` on the wrapper is what actually gets returned to callers.
    base_retriever = db.as_retriever(search_kwargs={"k": len(GOLDEN_CORPUS)})
    return TopicBoostedRetriever(base_retriever=base_retriever, top_k=top_k, boost_factor=boost_factor)


def run_eval(top_k: int = 3, boost_factor: float = 0.5) -> dict:
    tmp_dir = tempfile.mkdtemp(prefix="rag_eval_chroma_")
    try:
        retriever = build_eval_retriever(tmp_dir, top_k=top_k, boost_factor=boost_factor)

        results = []
        for case in GOLDEN_QUERIES:
            docs = retriever.invoke(case["query"])
            retrieved_ids = [_doc_id(d) for d in docs]
            retrieved_topics = set()
            for d in docs:
                retrieved_topics |= set(d.metadata.get("doc_topics", []))

            topic_hit = bool(case["expect_topics"] & retrieved_topics)
            doc_hit = case.get("expect_doc_id") in retrieved_ids if case.get("expect_doc_id") else None

            results.append({
                "query": case["query"],
                "expected_topics": sorted(case["expect_topics"]),
                "expected_doc_id": case.get("expect_doc_id"),
                "retrieved_ids": retrieved_ids,
                "retrieved_topics": sorted(retrieved_topics),
                "topic_hit": topic_hit,
                "doc_hit": doc_hit,
                "pass": topic_hit and (doc_hit is not False),
            })

        n = len(results)
        topic_hits = sum(r["topic_hit"] for r in results)
        doc_hits = sum(1 for r in results if r["doc_hit"])
        overall_pass = sum(r["pass"] for r in results)

        return {
            "top_k": top_k,
            "boost_factor": boost_factor,
            "n_queries": n,
            "topic_hit_rate": topic_hits / n,
            "doc_hit_rate": doc_hits / n,
            "overall_hit_rate": overall_pass / n,
            "results": results,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _doc_id(doc) -> str | None:
    # The Chroma retriever doesn't surface the original document id in
    # metadata, so match retrieved passages back to the golden corpus by
    # their (unique, verbatim) text instead.
    return _ID_BY_TEXT.get(doc.page_content)


def print_report(report: dict) -> None:
    print(f"Retrieval eval — top_k={report['top_k']} boost_factor={report['boost_factor']}")
    print(f"{'='*70}")
    for r in report["results"]:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[{mark}] {r['query']}")
        print(f"       expected topics : {r['expected_topics']}")
        print(f"       retrieved topics: {r['retrieved_topics']}")
        if r["expected_doc_id"]:
            print(f"       expected doc '{r['expected_doc_id']}' in top-k: {r['doc_hit']} (got {r['retrieved_ids']})")
        print()
    print(f"{'='*70}")
    print(f"Topic hit rate:   {report['topic_hit_rate']:.0%}  ({report['n_queries']} queries)")
    print(f"Doc hit rate:     {report['doc_hit_rate']:.0%}")
    print(f"Overall hit rate: {report['overall_hit_rate']:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--boost-factor", type=float, default=0.5)
    parser.add_argument("--min-hit-rate", type=float, default=0.7, help="Exit non-zero if overall hit rate is below this")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text report")
    args = parser.parse_args()

    if args.json:
        # embeddings.py / vector_store.py print model-loading status to
        # stdout — redirect that noise to stderr so stdout is clean JSON.
        with contextlib.redirect_stdout(sys.stderr):
            report = run_eval(top_k=args.top_k, boost_factor=args.boost_factor)
        print(json.dumps(report, indent=2))
    else:
        report = run_eval(top_k=args.top_k, boost_factor=args.boost_factor)
        print_report(report)

    if report["overall_hit_rate"] < args.min_hit_rate:
        print(
            f"\nFAIL: overall hit rate {report['overall_hit_rate']:.0%} "
            f"is below the --min-hit-rate threshold ({args.min_hit_rate:.0%})",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
