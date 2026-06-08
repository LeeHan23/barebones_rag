"""
RAG orchestration — stripped of CLaRa, patient auth, and multi-tenancy.
Exposes rag_stream() which yields LLM tokens for StreamingResponse.
"""
from typing import Generator

from chain_factory import create_conversational_chain
from llm import get_direct_llm_response


def identify_target_disease(question: str) -> str:
    """Lightweight LLM call to extract the primary health topic from the question."""
    prompt = (
        f'Identify the primary health condition in this question. '
        f'If a specific condition is mentioned (e.g. "Type 2 Diabetes", "Hypertension", "CKD"), '
        f'return only that name. If none is mentioned, return "general health and wellness".\n'
        f'Question: "{question}"\nCondition:'
    )
    result = get_direct_llm_response(prompt)
    print(f"[RAG] target disease: {result or 'general health and wellness'}")
    return result or "general health and wellness"


def rag_stream(question: str, session_id: str) -> Generator[str, None, None]:
    """
    Embed → retrieve → stream tokens from the LCEL chain.
    Yields raw text chunks suitable for SSE or plain StreamingResponse.
    """
    target_disease = identify_target_disease(question)
    chain = create_conversational_chain(target_disease)

    for chunk in chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk
