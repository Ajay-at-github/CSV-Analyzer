import asyncio

import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory

from src.llm_manager import (
    ask_question,
    build_chain,
    get_llm,
)
from src.retrieve import retriever_func
from src.session_manager import (
    add_message,
    clear_messages,
    create_chat,
    create_chat_if_needed,
    dataset_exists,
    get_chat,
    get_current_chat,
    get_current_chat_id,
    get_current_dataset,
    get_dataset_chats,
    get_file_hash,
    get_messages,
    initialize_session,
    rename_current_chat,
    save_dataset,
    set_current_chat,
)


def chat(
    temperature,
    model_name,
    user_api_key,
    uploaded_file,
):
    initialize_session()

    st.header("💬 Chat with CSV")

    if uploaded_file is None:
        st.info("Please upload a CSV file from the sidebar.")
        return

    dataset_id = get_file_hash(uploaded_file)

    if not dataset_exists(dataset_id):

        with st.spinner("Processing CSV..."):

            df, retriever, vectorstore = retriever_func(
                uploaded_file
            )

        save_dataset(
            dataset_id=dataset_id,
            file_name=uploaded_file.name,
            dataframe=df,
            retriever=retriever,
            vectorstore=vectorstore,
        )

    create_chat_if_needed(dataset_id)

    dataset = get_current_dataset()

    df = dataset["dataframe"]
    vectorstore = dataset["vectorstore"]

    st.sidebar.divider()
    st.sidebar.subheader("💬 Chat History")

    chats = get_dataset_chats(dataset_id)

    for chat_session in chats:

        if st.sidebar.button(
            chat_session["title"],
            key=chat_session["id"],
            use_container_width=True,
        ):
            set_current_chat(chat_session["id"])
            st.rerun()

    if st.sidebar.button(
        "➕ New Chat",
        use_container_width=True,
    ):
        create_chat(dataset_id)
        st.rerun()

    if st.sidebar.button(
        "🗑 Reset Current Chat",
        use_container_width=True,
    ):
        clear_messages()
        st.rerun()

    st.subheader("📄 CSV Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        int(df.isna().sum().sum()),
    )

    llm = get_llm(
        model_name=model_name,
        temperature=temperature,
        api_key=user_api_key,
    )

    def history_callback(
        session_id: str,
    ) -> BaseChatMessageHistory:
        return get_chat(session_id)["history"]

    chain = build_chain(
        llm,
        history_callback,
    )

    for message in get_messages():
        st.chat_message(
            message["role"]
        ).write(
            message["content"]
        )

    async def chat_loop():

        question = st.chat_input(
            "Ask anything about your CSV..."
        )

        if not question:
            return

        if not user_api_key:
            st.error(
                "Please enter your Groq API key."
            )
            return

        if (
            get_current_chat()["title"]
            == "New Chat"
        ):
            rename_current_chat(question)

        add_message(
            "user",
            question,
        )

        st.chat_message("user").write(question)

        docs = vectorstore.similarity_search(
            question,
            k=6,
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        with st.chat_message("assistant"):

            placeholder = st.empty()

            result = ask_question(
                chain=chain,
                session_id=get_current_chat_id(),
                context=context,
                question=question,
            )

            placeholder.markdown(
                result.content
            )

        add_message(
            "assistant",
            result.content,
        )

    asyncio.run(chat_loop())