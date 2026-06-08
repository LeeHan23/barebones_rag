# Barebones RAG

A minimal, self-hosted Retrieval-Augmented-Generation chatbot: FastAPI +
ChromaDB + a swappable local/cloud LLM, wrapped in a conversational chain with
a configurable persona (ships with an ADIME nutrition-assistant persona for a
Malaysian multicultural setting) and topic-aware retrieval re-ranking.

> For a deep dive into the architecture, data flow, and how to extend each
> layer (personas, document loaders, retrievers, LLM/embedding backends), see
> **[FRAMEWORK.md](FRAMEWORK.md)**.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (default), **or** an OpenAI /
  OpenAI-compatible API key
- ~2GB free for the default embedding model (`BAAI/bge-m3`, downloaded on first run)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — at minimum, make sure OLLAMA_MODEL matches a model you've pulled:
ollama pull llama3.2:3b

# 3. Run the API
uvicorn app:app --reload
```

The server starts at `http://localhost:8000`. On first request it downloads/loads
the embedding model and opens (or creates) the Chroma collection at `CHROMA_PATH`.

## Configuration

All configuration lives in `.env` (see `.env.example` for the full list with
defaults):

| Variable | Purpose |
|---|---|
| `USE_OLLAMA` | `true` for local Ollama, `false` for OpenAI / OpenAI-compatible |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | Local model tag + server URL (must be `ollama pull`-ed first) |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | Cloud LLM config (read only when `USE_OLLAMA=false`) |
| `EMBEDDING_MODEL` / `EMBEDDING_ADAPTER_PATH` | Embedding model + optional LoRA adapter |
| `CHROMA_PATH` | Where the vector store is persisted on disk |

See [FRAMEWORK.md §3](FRAMEWORK.md#3-configuration-env) for the full reference
and known gotchas (e.g. `OLLAMA_MODEL` mismatches surfacing as a 404 deep in
the LangChain stack).

## Usage

### Ingest documents

Via the API (`.pdf`, `.docx`, `.txt`, `.md`):

```bash
curl -X POST http://localhost:8000/ingest -F "file=@guide.pdf"
```

Or in bulk from the CLI:

```bash
python ingest.py /path/to/docs/                 # file or directory
python ingest.py /path/to/docs/ --chunk-size 600 --overlap 80
```

Both paths are idempotent — re-ingesting unchanged content adds nothing
(`chunks_skipped` reflects content already in the store).

### Chat

Streaming (Server-Sent Events):

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I eat if I have high blood pressure?"}'
```

Non-streaming (single JSON response):

```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I eat if I have high blood pressure?", "session_id": "my-session"}'
```

Pass the same `session_id` across requests to continue a conversation with
memory — omit it and the server generates one (returned in the `X-Session-ID`
response header / JSON body).

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "chunks_in_store": 42}
```

## Testing & evaluation

```bash
pytest                      # fast unit/integration suite (~2s, no model loads)
python eval_retrieval.py    # retrieval-quality regression check against a real
                            # embedding model + a small golden Q&A corpus
```

See [FRAMEWORK.md §6](FRAMEWORK.md#6-testing--evaluation) for what each suite
covers and when to run the retrieval eval.

## Project structure

```
app.py             FastAPI app — /health, /chat, /chat/sync, /ingest
ingestion.py       Shared text extraction + chunking (used by app.py & ingest.py)
ingest.py          CLI for bulk document ingestion
embeddings.py      Embedding-model singleton (base model or LoRA adapter)
vector_store.py    ChromaDB wiring, topic taxonomy, TopicBoostedRetriever
llm.py             Chat-model factory (Ollama or OpenAI)
chain_factory.py   LCEL conversational chain — persona + retrieval + memory
rag.py             Orchestration glue — exposes rag_stream() for the API
eval_retrieval.py  Retrieval-quality evaluation harness
tests/             pytest suite (mocked — no heavy model loads)
data/chroma/       Persisted vector store (gitignored contents may vary)
```

For everything else — architecture diagrams, data-flow walkthroughs, the topic
boost's ranking characteristics, and extension recipes — see
**[FRAMEWORK.md](FRAMEWORK.md)**.
