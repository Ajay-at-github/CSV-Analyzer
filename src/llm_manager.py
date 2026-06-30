import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


# ==========================================================
# System Prompt
# ==========================================================

SYSTEM_PROMPT = """
You are a CSV analysis assistant.

Your job is to answer questions ONLY using the provided CSV context.

Guidelines:

- Answer only from the supplied context.
- If the answer is not present in the CSV, clearly say:
  "I couldn't find that information in the uploaded CSV."
- Do not make assumptions.
- Do not fabricate values.
- If calculations are requested, perform them only using the supplied data.

Context:
{context}
"""


# ==========================================================
# Cached LLM
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_llm(
    model_name: str,
    temperature: float,
    api_key: str,
):
    """
    Returns a cached Groq LLM client.
    """

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )


# ==========================================================
# Prompt
# ==========================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "{input}",
        ),
    ]
)


# ==========================================================
# Runnable Chain
# ==========================================================

def build_chain(
    llm,
    history_callback,
):
    """
    Builds a RunnableWithMessageHistory.
    """

    runnable = PROMPT | llm

    return RunnableWithMessageHistory(
        runnable=runnable,
        get_session_history=history_callback,
        input_messages_key="input",
        history_messages_key="history",
    )


# ==========================================================
# Invoke Helper
# ==========================================================

def ask_question(
    chain,
    session_id: str,
    context: str,
    question: str,
):
    """
    Invokes the LangChain runnable and returns the response.
    """

    return chain.invoke(
        {
            "context": context,
            "input": question,
        },
        config={
            "configurable": {
                "session_id": session_id,
            }
        },
    )