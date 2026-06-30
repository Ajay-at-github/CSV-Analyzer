import os
import streamlit as st
from dotenv import load_dotenv

from src.chat import chat

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="CSV Analyzer",
    page_icon="🧠",
    layout="wide"
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -----------------------------------------------------
# Constants
# -----------------------------------------------------

MODEL_OPTIONS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

TEMPERATURE = 0.1
TOP_P = 1.0


# -----------------------------------------------------
# Header
# -----------------------------------------------------

def render_header():

    st.markdown(
        """
        <div style="text-align:center;">
            <h1>🧠 CSV Analyzer</h1>
            <h4>⚡ Interact with and summarize CSV files using LLMs</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

def render_sidebar():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        api_key = st.sidebar.text_input(
            "#### Enter Groq API Key",
            type="password",
            placeholder="Paste your Groq API key",
        )

        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

    st.sidebar.divider()

    model_name = st.sidebar.selectbox(
        "Model",
        MODEL_OPTIONS,
    )

    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="main_upload",
    )

    return api_key, model_name, uploaded_file


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    render_header()

    api_key, model_name, uploaded_file = render_sidebar()

    tab_chat, tab_summary = st.tabs(
        [
            "💬 Chat with CSV",
            "📝 Summarize CSV",
        ]
    )

    with tab_chat:

        chat(
            temperature=TEMPERATURE,
            model_name=model_name,
            user_api_key=api_key,
            uploaded_file=uploaded_file,
        )

    with tab_summary:

        summary(
            model_name=model_name,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            user_api_key=api_key,
            uploaded_file=uploaded_file,
        )


if __name__ == "__main__":
    main()
