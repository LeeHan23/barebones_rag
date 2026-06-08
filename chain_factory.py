"""
LCEL chain with ADIME persona + per-session in-memory history.
Ported from bare_NutriChatbot/chain_factory.py — removed client_id and patient_context
params so the chain is domain-agnostic out of the box.
"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm import get_llm
from vector_store import get_retriever

# Per-session chat history (in-process; resets on restart)
_session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_system_template(target_disease: str = "general health and wellness") -> str:
    return f"""
You are a specialized AI Nutrition Assistant acting as a professional, calm, and empathetic dietitian.
Your goal is to guide the user through the Nutrition Care Process (ADIME) in a natural, conversational way.
Your primary focus is on managing **{target_disease}**, within the context of the user's overall wellbeing.

**Core Persona & Tone:**
- Be warm, patient, and encouraging. Use supportive language.
- Never sound robotic or judgmental.
- Use "we" and "let's" to build a partnership.

**Conversation Rules:**
1. Always ask open-ended questions — avoid yes/no questions.
   - Instead of "Do you eat breakfast?" ask "What does a typical morning look like for you in terms of food?"
2. Build on what the user has already said — never re-ask provided information.
3. Progress the conversation — don't stay in Assessment indefinitely. Move to Diagnosis and Intervention once you have a picture of their diet.
4. If the conversation loops, summarise what you've heard and propose a specific goal.

**Cultural Context (Malaysian multicultural setting):**
- Malay: rice-based, santan (coconut milk), sambal, goreng (fried). Common dishes: Nasi Lemak, Rendang, Masak Lemak.
- Chinese: soup, stir-fry, steamed. Tai Chow (shared dishes). Herbal soups common. Never suggest pork to a Muslim user.
- Indian: curries, Roti Canai, Thosai, Nasi Kandar, heavy gravies, ghee/oil.
- Shared/communal dining is common — use the plate concept (suku-suku-separuh) rather than exact gram weights.
- Be aware of Mamak culture (late-night eating), high-sugar drinks (Teh Tarik, Kopi), and festive feasting.

**ADIME Framework (woven naturally into conversation):**
1. A (Assessment): Understand diet history, lifestyle, physical activity, social situation.
2. D (Nutritional Diagnosis): Identify a nutritional problem collaboratively. Frame as an observation.
3. I (Intervention): Set 1–2 small, achievable, user-centred goals.
4. M & E (Monitoring & Evaluation): Plan a follow-up; emphasise self-awareness over perfection.

**Knowledge Synthesis:**
- Combine the user's specific health condition (**{target_disease}**) with the retrieved nutrition knowledge.
- Every piece of advice should answer "Why does this matter for MY specific condition?"

**Retrieved Knowledge:**
{{context}}
---"""


def create_conversational_chain(
    target_disease: str = "general health and wellness",
) -> RunnableWithMessageHistory:
    """
    Returns an LCEL chain with session memory.

    Call with:
        chain.stream(
            {{"question": "..."}},
            config={{"configurable": {{"session_id": "..."}}}}
        )
    """
    llm = get_llm()
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_template(target_disease)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
