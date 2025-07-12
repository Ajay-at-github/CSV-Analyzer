import streamlit as st
import os
from dotenv import load_dotenv

from src.chat import chat
from src.summary import summary
from src.retrieve import home_page

load_dotenv()

st.set_page_config(page_title="CSV-Analyzer", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🧠 CSV-Analyzer</h1>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            <h4>⚡️ Interacting and Summarizing CSV Files!</h4>
        </div>
        """, unsafe_allow_html=True)

    global user_api_key
    user_api_key = os.environ.get("GROQ_API_KEY")

    if user_api_key:
        st.success("API key loaded from .env", icon="🚀")
    else:
        user_api_key = st.sidebar.text_input(
            label="#### Enter Groq API key 👇",
            placeholder="Paste your Groq API key",
            type="password",
            key="groq_api_key"
        )
        if user_api_key:
            st.sidebar.success("API key loaded", icon="🚀")
            os.environ["GROQ_API_KEY"] = user_api_key 

    MODEL_OPTIONS = ["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it"]
    model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
    top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9, 0.01)

    functions = [
        "home",
        "Chat with CSV",
        "Summarize CSV",
    ]

    selected_function = st.selectbox("Select a functionality", functions)
    if selected_function == "home":
        home_page()
    elif selected_function == "Chat with CSV":
        chat(temperature=temperature, model_name=model_name, user_api_key=user_api_key)
    elif selected_function == "Summarize CSV":
        summary(model_name=model_name, temperature=temperature, top_p=top_p, user_api_key=user_api_key)
    else:
        st.warning("You haven't selected any AI Functionality!!")

if __name__ == "__main__":
    main()