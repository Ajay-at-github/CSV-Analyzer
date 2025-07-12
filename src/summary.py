import streamlit as st
import os, tempfile
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain.chains.summarize import load_summarize_chain

def summary(model_name, temperature, top_p, user_api_key):
    st.write("# Summary of CSV")
    st.write("Upload your document here:")
    uploaded_file = st.file_uploader("Upload source document", type="csv", label_visibility="collapsed")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()
            texts = text_splitter.split_documents(data)
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
            texts = text_splitter.split_documents(data)

        os.remove(tmp_file_path)
        gen_sum = st.button("Generate Summary")
        if gen_sum:
            llm = ChatGroq(model_name=model_name, temperature=temperature, groq_api_key=user_api_key)
            chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                return_intermediate_steps=True,
                input_key="input_documents",
                output_key="output_text",
            )
            result = chain({"input_documents": texts}, return_only_outputs=True)
            st.success(result["output_text"])