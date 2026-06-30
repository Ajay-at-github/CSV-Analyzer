import pandas as pd
import streamlit as st

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================================
# Embedding Model
# ==========================================================

@st.cache_resource
def get_embedding_model():
    """
    Loads the embedding model only once.
    """

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# ==========================================================
# Read CSV
# ==========================================================

def load_dataframe(uploaded_file):
    """
    Reads the uploaded CSV into a pandas DataFrame.
    """

    uploaded_file.seek(0)

    try:
        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip",
        )

    except Exception:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            encoding="cp1252",
            on_bad_lines="skip",
        )

    uploaded_file.seek(0)

    return df


# ==========================================================
# DataFrame -> Documents
# ==========================================================

def dataframe_to_documents(df):
    """
    Converts every DataFrame row into a LangChain Document.
    """

    documents = []

    for _, row in df.iterrows():

        content = "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
        )

        documents.append(
            Document(
                page_content=content
            )
        )

    return documents


# ==========================================================
# Build Retriever
# ==========================================================

def retriever_func(uploaded_file):
    """
    Processes the uploaded CSV and returns:
        - DataFrame
        - Retriever
        - FAISS Vector Store
    """

    # Read CSV
    df = load_dataframe(uploaded_file)

    # Convert DataFrame to LangChain Documents
    documents = dataframe_to_documents(df)

    # Split Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    split_documents = splitter.split_documents(documents)

    # Load embedding model
    embeddings = get_embedding_model()

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        split_documents,
        embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,
        },
    )

    return df, retriever, vectorstore