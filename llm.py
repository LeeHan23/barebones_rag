"""
LLM factory — Ollama (local) or OpenAI (cloud), toggled via USE_OLLAMA env var.
Stripped CLaRa/compress paths from bare_NutriChatbot/llm.py.
"""
import os

from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() in ("true", "1", "yes")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

if not USE_OLLAMA and not OPENAI_API_KEY:
    raise EnvironmentError(
        "No LLM configured. Set USE_OLLAMA=true or provide OPENAI_API_KEY in .env"
    )


def get_llm():
    """LangChain chat model — used by the LCEL chain in chain_factory.py."""
    if USE_OLLAMA:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=512,
            keep_alive=-1,
            timeout=300,
        )
    else:
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model_name=OPENAI_MODEL,
            temperature=0.3,
            max_tokens=512,
            openai_api_key=OPENAI_API_KEY,
        )
        if OPENAI_BASE_URL:
            kwargs["openai_api_base"] = OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)


def get_direct_llm_response(prompt: str) -> str:
    """Single-shot LLM call for lightweight tasks (e.g. disease identification)."""
    try:
        return get_llm().invoke(prompt).content.strip()
    except Exception as e:
        print(f"[LLM error] {e}")
        return ""
