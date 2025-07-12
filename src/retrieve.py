import streamlit as st
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader

def home_page():
    st.write("""Select any one feature from above sliderbox: \n
    1. Chat with CSV \n
    2. Summarize CSV""")

@st.cache_resource()
def retriever_func(uploaded_file):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(data)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    else:
        st.info("Please upload CSV documents to continue.")
        st.stop()
    return retriever, vectorstore