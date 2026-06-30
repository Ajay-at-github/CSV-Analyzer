import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

from src.llm_manager import get_llm
from src.retrieve import retriever_func
from src.session_manager import (
    initialize_session,
    get_file_hash,
    dataset_exists,
    save_dataset,
    get_dataset,
)


def summary(
    model_name,
    temperature,
    top_p,
    user_api_key,
    uploaded_file,
):
    initialize_session()

    st.header("📝 Summarize CSV")

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

    dataset = get_dataset(dataset_id)

    df = dataset["dataframe"]

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

    if st.button(
        "Generate Summary",
        use_container_width=True,
    ):

        if not user_api_key:
            st.error("Please enter your Groq API key.")
            return

        with st.spinner("Generating summary..."):

            documents = [
                Document(
                    page_content="\n".join(
                        f"{column}: {value}"
                        for column, value in row.items()
                    )
                )
                for _, row in df.iterrows()
            ]

            llm = get_llm(
                model_name=model_name,
                temperature=temperature,
                api_key=user_api_key,
            )

            chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                input_key="input_documents",
                output_key="output_text",
                return_intermediate_steps=False,
            )

            result = chain.invoke(
                {
                    "input_documents": documents
                }
            )

        st.subheader("📄 Summary")

        st.markdown(
            f"""
            <div style="
            padding:20px;
            border-radius:10px;
            border:1px solid #444;
            background:#1E1E1E;
            ">
            {result["output_text"]}
            </div>
            """,
            unsafe_allow_html=True,
        )                                                           

        st.download_button(
            label="⬇ Download Summary",
            data=result["output_text"],
            file_name="summary.txt",
            mime="text/plain",
        )