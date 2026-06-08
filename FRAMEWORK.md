# Barebones RAG — Framework Guide

A minimal, self-hosted Retrieval-Augmented-Generation chatbot: FastAPI + ChromaDB
+ a swappable local/cloud LLM, wrapped in an LCEL conversational chain with a
domain persona (currently an ADIME nutrition assistant) and topic-aware
retrieval re-ranking.

This document explains how the pieces fit together, how data flows through the
system, how to configure it, and how to extend each layer.

---

## 1. Component map

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
│   app.py    │────▶│   rag.py     │────▶│  chain_factory.py  │
│  (FastAPI)  │     │ orchestration│     │  (LCEL + persona)  │
└─────────────┘     └──────────────┘     └────────────────────┘
       │                    │                       │      │
       │                    ▼                       ▼      ▼
       │             ┌────────────┐         ┌───────────┐ ┌──────────┐
       │             │   llm.py   │         │vector_store│ │  llm.py  │
       │             │ (disease   │         │ .py        │ │ (chat    │
       │             │  ID call)  │         │ (retriever)│ │  model)  │
       │             └────────────┘         └───────────┘ └──────────┘
       │                                           │
       ▼                                           ▼
┌─────────────┐                            ┌──────────────┐
│ ingestion.py│◀───────────────────────────│ embeddings.py│
│ (extract +  │   shared by app.py &       │ (singleton   │
│  chunk)     │   ingest.py CLI            │  embed model)│
└─────────────┘                            └──────────────┘
       │
       ▼
┌─────────────────────┐
│  ChromaDB at        │
│  CHROMA_PATH        │ ◀── persisted on disk, single "knowledge" collection
└─────────────────────┘
```

| File | Responsibility |
|---|---|
| `app.py` | FastAPI app: `/health`, `/chat`, `/chat/sync`, `/ingest`. Validates input, streams SSE. |
| `ingestion.py` | Shared text-extraction (`pdf`/`docx`/`txt`/`md`) and word-based chunking. Used by both `app.py`'s `/ingest` endpoint and the `ingest.py` CLI so the two paths can never drift. |
| `ingest.py` | CLI for bulk-loading a file or directory into the vector store. |
| `embeddings.py` | Singleton embedding-function factory — base `BAAI/bge-m3` or an optional LoRA adapter. |
| `vector_store.py` | ChromaDB wiring + `TOPIC_HINTS` taxonomy + `TopicBoostedRetriever` (re-ranks by topic overlap). |
| `llm.py` | Chat-model factory — Ollama (local) or OpenAI (cloud), toggled by `USE_OLLAMA`. Also exposes a single-shot `get_direct_llm_response`. |
| `chain_factory.py` | Builds the LCEL conversational chain: persona system prompt + retrieved context + per-session chat history. |
| `rag.py` | Orchestration glue: identifies the user's health topic, builds the chain, and exposes `rag_stream()` for the API to consume token-by-token. |

---

## 2. Data flow

### 2a. Ingestion (`POST /ingest` or `python ingest.py <path>`)

1. Read raw file bytes; reject unsupported extensions (`SUPPORTED_EXTENSIONS` in `ingestion.py`).
2. `extract_text(ext, bytes)` — pulls plain text out of PDF/DOCX/plain text.
3. `chunk_text(text, size=800, overlap=100)` — splits into overlapping word-count windows.
4. `sha256(file_bytes)[:16]` becomes the file's content hash; chunk IDs are `f"{hash}_{i}"`.
   This makes ingestion **idempotent** — re-ingesting an unchanged file adds nothing
   (`chunks_skipped` reflects IDs Chroma already has).
5. New chunks are embedded (via `embeddings.get_embedding_function()`) and written
   to the single `"knowledge"` Chroma collection at `CHROMA_PATH`.

> Chunks ingested through this path carry **no `doc_topics` metadata** — the
> topic boost (§4) only activates for chunks that have it. To get boosted
> retrieval, either pre-tag your corpus before ingestion (see §5.2) or accept
> that the boost is a no-op for untagged chunks (it falls back to embedding rank).

### 2b. Chat (`POST /chat` streaming, `POST /chat/sync` JSON)

1. `app.py` validates the question is non-empty and resolves a `session_id`
   (generates a UUID if the caller didn't supply one).
2. `rag.rag_stream(question, session_id)`:
   a. **Disease identification** — a single-shot LLM call (`identify_target_disease`)
      extracts the primary health condition from the question (e.g. "Type 2 Diabetes"),
      defaulting to `"general health and wellness"`.
   b. **Chain construction** — `chain_factory.create_conversational_chain(target_disease)`
      builds a fresh LCEL chain whose system prompt is parameterised by that condition.
   c. **Retrieval** — inside the chain, `retriever.invoke(question)` fetches a
      candidate pool from Chroma, then `TopicBoostedRetriever` re-ranks it (§4)
      and the top 5 chunks are joined into a `context` string.
   d. **Generation** — the prompt (system persona + `context` + chat history +
      question) is sent to the LLM (`llm.get_llm()`), and tokens stream back
      through `StrOutputParser`.
   e. **Memory** — `RunnableWithMessageHistory` records the turn in an
      in-process `InMemoryChatMessageHistory` keyed by `session_id`
      (resets on restart — see §6 for swapping in persistent memory).
3. `app.py` wraps each token as an SSE frame (`data: <chunk>\n\n`) for `/chat`,
   or joins them into a single JSON `answer` for `/chat/sync`.

**Note on streaming + memory**: `RunnableWithMessageHistory` only persists the
turn to history once the *generator is fully consumed*. If a client disconnects
or truncates the SSE stream early, that turn is silently dropped from memory —
this is standard LangChain behaviour, not a bug, but worth knowing when debugging
"the bot forgot what we discussed."

---

## 3. Configuration (`.env`)

| Variable | Purpose | Default |
|---|---|---|
| `USE_OLLAMA` | `true` → local Ollama, `false` → OpenAI-compatible API | `true` |
| `OLLAMA_MODEL` | Ollama model tag (must be `ollama pull`-ed first!) | `llama3.2:3b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | Cloud LLM config (only read when `USE_OLLAMA=false`; `OPENAI_BASE_URL` also lets you point at any OpenAI-compatible endpoint, e.g. LM Studio) | — |
| `EMBEDDING_MODEL` | HuggingFace embedding model name | `BAAI/bge-m3` |
| `EMBEDDING_ADAPTER_PATH` | Optional path to a LoRA adapter (see `embeddings._load_lora_embedding`) | — |
| `EMBEDDING_DEVICE` | Device for the LoRA path (`cpu`/`cuda`) | `cpu` |
| `CHROMA_PATH` | Where the persisted Chroma collection lives on disk | `/mnt/ext/barebones_rag/data/chroma` |

**Gotcha**: `OLLAMA_MODEL` must exactly match a tag you've pulled
(`ollama list`). A mismatch surfaces as a 500 with
`OllamaEndpointNotFoundError: ... status code 404` deep in the LangChain stack —
easy to mistake for an app bug.

---

## 4. The topic-boost retriever

`vector_store.py` defines `TOPIC_HINTS`: a dict mapping query phrases (English
and Bahasa Malaysia) to sets of topic tags. `detect_query_topics(query)` returns
the union of tags for every phrase that appears in the query.

`TopicBoostedRetriever` wraps a base similarity-search retriever:

1. Fetch a wide candidate pool (`k=15` by default, see `get_retriever`).
2. Compute `query_topics` from the query text **and** any `patient_conditions`.
3. Score each candidate: `base_score (1/(rank+1)) + boost_factor * topic_overlap_ratio`.
4. Sort by `(-score, original_rank)` and return the top `top_k` (default 5).

**Important characteristic** (covered by `tests/test_vector_store.py`): with the
default `boost_factor=0.5`, a perfectly-matching document can at best *tie* an
unrelated rank-0 document (`0.5 base + 0.5 boost == 1.0 == unboosted rank-0
score`), and ties favour the lower rank index. So **the boost re-orders
candidates below the top slot — it cannot dethrone an irrelevant #1 embedding
match**. If you need the boost to be able to override embedding rank entirely,
raise `boost_factor` above `0.5`.

Chunks ingested without `doc_topics` metadata simply keep their embedding rank
(the boost is additive, never exclusionary).

---

## 5. Extending the system

### 5.1 Swap the LLM backend
Edit `.env`: set `USE_OLLAMA=false` and provide `OPENAI_API_KEY` (or point
`OPENAI_BASE_URL` at any OpenAI-compatible server — LM Studio, vLLM, etc.).
No code changes needed; `llm.get_llm()` handles both paths.

### 5.2 Add topic-tagged documents
`db.add_texts` accepts a `metadatas=[{"doc_topics": [...]}, ...]` argument
(see `eval_retrieval.py::build_eval_retriever` for a working example). The
current `/ingest` endpoint and CLI don't populate this — if you want boosted
retrieval on bulk-ingested content, either:
- pre-process your corpus to attach `doc_topics` per chunk before calling
  `db.add_texts`, or
- extend `_extract_text`/ingestion to auto-tag chunks (e.g. run
  `detect_query_topics` over each chunk's own text and store the result).

### 5.3 Extend the topic taxonomy
Add entries to `TOPIC_HINTS` in `vector_store.py` — both the matching phrase
(lowercase; surrounded-by-spaces matching means multi-word phrases like
`" fat "` or `" ckd "` need leading/trailing spaces) and the topic tag set.
No other code changes required.

### 5.4 Change the persona / system prompt
The entire persona (the ADIME nutrition-assistant framing, Malaysian cultural
context, conversational rules) lives in `chain_factory.get_system_template()`.
To adapt this "barebones" RAG to a different domain, replace that template —
the `{context}` placeholder is the only contract the chain relies on. The
`target_disease` parameter is domain-specific to the nutrition use case; for a
general-purpose assistant you can drop it (and the `identify_target_disease`
call in `rag.py`) entirely.

### 5.5 Add new document types to ingestion
Add the extension to `SUPPORTED_EXTENSIONS` in `ingestion.py` and a branch in
`extract_text()`. Both `/ingest` and the CLI pick it up automatically since
they share this module.

### 5.6 Persistent chat memory
`chain_factory._session_store` is an in-process dict — it resets on restart and
won't work across multiple server instances. To persist sessions, swap
`InMemoryChatMessageHistory` for a LangChain history backend that writes to
Redis/Postgres/SQLite (e.g. `RedisChatMessageHistory`), keeping the
`get_session_history(session_id)` contract intact.

### 5.7 Embedding model / LoRA adapters
Set `EMBEDDING_MODEL` to any sentence-embedding-capable HuggingFace model, or
`EMBEDDING_ADAPTER_PATH` to a directory containing a `nutribot_adapter_config.json`
(base model name, pooling mode, max sequence length) plus PEFT adapter weights —
see `embeddings._load_lora_embedding` for the exact contract.

---

## 6. Testing & evaluation

```
tests/
├── conftest.py          # FakeChroma + mocked rag_stream — no model loads
├── test_ingestion.py    # chunk_text / extract_text unit tests
├── test_vector_store.py # detect_query_topics + TopicBoostedRetriever ranking logic
└── test_api.py          # FastAPI endpoint behaviour (health/chat/ingest)

eval_retrieval.py        # end-to-end retrieval quality check against a real
                         # embedding model + a small golden Q&A corpus
```

- **Unit/integration tests** (`pytest`) run in ~2 seconds with no model loads —
  the vector store and RAG chain are mocked (`tests/conftest.py`). These guard
  routing, validation, chunking math, and the topic-boost ranking *logic*.
- **Retrieval eval** (`python eval_retrieval.py`) loads the *real* embedding
  model, builds a throwaway Chroma collection from a 7-document golden corpus
  spanning the major topic areas (hypertension, diabetes, CKD, cholesterol,
  weight, smoking, general nutrition), and checks that each golden query
  retrieves the expected document/topics in its top-k. Run this whenever you:
  - change the embedding model or adapter
  - tune `top_k` / `boost_factor`
  - edit `TOPIC_HINTS`

  ```
  python eval_retrieval.py                 # human-readable report
  python eval_retrieval.py --json          # machine-readable, for CI
  python eval_retrieval.py --min-hit-rate 0.8   # exits non-zero below threshold
  ```

Run everything:
```
pytest                      # fast unit/integration suite
python eval_retrieval.py    # slower, model-backed retrieval regression check
```

---

## 7. Known rough edges (as of this audit)

- **`OLLAMA_MODEL` default mismatch**: `.env.example` suggested `llama3`, which
  isn't pulled on this machine (only `llama3.2:3b` and several fine-tuned
  variants are). `.env` here has been corrected to `llama3.2:3b`.
- **`requirements.txt` was stale**: it pinned `langchain==0.1.20` / `langchain-community==0.0.38`,
  but the environment actually running the app has `langchain~=1.2`,
  `langchain-community~=0.4` (a major-version jump). The app works on the newer
  versions via `langchain_community` compatibility shims, but emits
  `LangChainDeprecationWarning`s for `HuggingFaceEmbeddings`, `Chroma`, and
  `ChatOllama`. `requirements.txt` now reflects the versions actually in use;
  migrating the three imports to `langchain-huggingface`/`langchain-chroma`/
  `langchain-ollama` would silence the warnings (not done here, to avoid
  changing working behaviour without your sign-off).
- **`.venv/` is an empty shell**: no `pyvenv.cfg`, no `bin/`, zero files — the
  app actually runs against the system/conda Python (`langchain` 1.2.x is
  installed there). Either remove `.venv/` or rebuild it
  (`python -m venv .venv && .venv/bin/pip install -r requirements.txt`).
- **Chunking infinite-loop bug (fixed)**: the original `_chunk`/`chunk` functions
  advanced `start += size - overlap`, which never advances (infinite loop) when
  `overlap >= size`. The shared `ingestion.chunk_text` now clamps the step to
  `max(1, size - overlap)`. (`tests/test_ingestion.py` regression-tests this.)
- **Bulk/API ingestion doesn't tag `doc_topics`**: chunks added via `/ingest`
  or `ingest.py` have no topic metadata, so `TopicBoostedRetriever` can't boost
  them — see §5.2 if you want that capability.
