import streamlit as st
import asyncio

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from src.retrieve import retriever_func

def chat(temperature, model_name, user_api_key):
    st.write("# Talk to CSV")
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    retriever, vectorstore = retriever_func(uploaded_file)
    llm = ChatGroq(model_name=model_name, temperature=temperature, groq_api_key=user_api_key)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    store = {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know. Context: {context}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    runnable = prompt | llm

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    async def chat_message():
        if prompt := st.chat_input():
            if not user_api_key:
                st.info("Please add your Groq API key to continue.")
                st.stop()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            contextt = vectorstore.similarity_search(prompt, k=6)
            context = "\n\n".join(doc.page_content for doc in contextt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                text_chunk = ""
                result = with_message_history.invoke(
                    {"context": context, "input": prompt},
                    config={"configurable": {"session_id": "abc123"}}
                )
                text_chunk += result.content
                message_placeholder.markdown(text_chunk)
                st.session_state.messages.append({"role": "assistant", "content": text_chunk})
        if reset:
            st.session_state["messages"] = []

    asyncio.run(chat_message())