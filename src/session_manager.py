import hashlib
import uuid

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory


# ==========================================================
# Session Initialization
# ==========================================================

def initialize_session():
    """
    Initializes all required Streamlit session state variables.
    """

    defaults = {
        "datasets": {},
        "chat_sessions": {},
        "current_chat_id": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==========================================================
# Dataset Management
# ==========================================================

def get_file_hash(uploaded_file):
    """
    Returns a unique MD5 hash for the uploaded CSV.
    """

    uploaded_file.seek(0)
    file_bytes = uploaded_file.getvalue()
    uploaded_file.seek(0)

    return hashlib.md5(file_bytes).hexdigest()


def dataset_exists(dataset_id):
    return dataset_id in st.session_state.datasets


def save_dataset(
    dataset_id,
    file_name,
    dataframe,
    retriever,
    vectorstore,
):
    """
    Stores a processed dataset.
    """

    st.session_state.datasets[dataset_id] = {
        "id": dataset_id,
        "file_name": file_name,
        "dataframe": dataframe,
        "retriever": retriever,
        "vectorstore": vectorstore,
    }


def get_dataset(dataset_id):
    return st.session_state.datasets.get(dataset_id)


def get_dataframe(dataset_id):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        return None

    return dataset["dataframe"]


def get_retriever(dataset_id):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        return None

    return dataset["retriever"]


def get_vectorstore(dataset_id):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        return None

    return dataset["vectorstore"]


def get_dataset_name(dataset_id):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        return None

    return dataset["file_name"]


# ==========================================================
# Chat Management
# ==========================================================

def create_chat(dataset_id):
    """
    Creates a new conversation for a dataset.
    """

    chat_id = str(uuid.uuid4())

    st.session_state.chat_sessions[chat_id] = {
        "id": chat_id,
        "title": "New Chat",
        "dataset_id": dataset_id,
        "messages": [
            {
                "role": "assistant",
                "content": "How can I help you?"
            }
        ],
        "history": ChatMessageHistory(),
    }

    st.session_state.current_chat_id = chat_id

    return chat_id


def chat_exists(chat_id):
    return chat_id in st.session_state.chat_sessions


def get_chat(chat_id):
    return st.session_state.chat_sessions.get(chat_id)


def get_all_chats():
    return st.session_state.chat_sessions


def delete_chat(chat_id):
    """
    Deletes a chat session.
    """

    if not chat_exists(chat_id):
        return

    del st.session_state.chat_sessions[chat_id]

    if st.session_state.current_chat_id == chat_id:

        remaining = list(st.session_state.chat_sessions.keys())

        st.session_state.current_chat_id = (
            remaining[0] if remaining else None
        )


# ==========================================================
# Current Chat
# ==========================================================

def set_current_chat(chat_id):
    if chat_exists(chat_id):
        st.session_state.current_chat_id = chat_id


def get_current_chat_id():
    return st.session_state.current_chat_id


def get_current_chat():

    chat_id = get_current_chat_id()

    if chat_id is None:
        return None

    return get_chat(chat_id)


def get_current_dataset_id():

    chat = get_current_chat()

    if chat is None:
        return None

    return chat["dataset_id"]


def get_current_dataset():

    dataset_id = get_current_dataset_id()

    if dataset_id is None:
        return None

    return get_dataset(dataset_id)


# ==========================================================
# Chat Messages
# ==========================================================

def get_messages():

    chat = get_current_chat()

    if chat is None:
        return []

    return chat["messages"]


def add_message(role, content):

    chat = get_current_chat()

    if chat is None:
        return

    chat["messages"].append(
        {
            "role": role,
            "content": content,
        }
    )


def clear_messages():

    chat = get_current_chat()

    if chat is None:
        return

    chat["messages"] = [
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]

    chat["history"] = ChatMessageHistory()


# ==========================================================
# LangChain History
# ==========================================================

def get_chat_history():

    chat = get_current_chat()

    if chat is None:
        return ChatMessageHistory()

    return chat["history"]


# ==========================================================
# Chat Title
# ==========================================================

def rename_current_chat(title):
    """
    Renames a chat only once.
    """

    chat = get_current_chat()

    if chat is None:
        return

    if chat["title"] == "New Chat":
        chat["title"] = title[:40]


# ==========================================================
# Dataset Chats
# ==========================================================

def get_dataset_chats(dataset_id):
    """
    Returns all chats for a dataset.
    """

    return [
        chat
        for chat in st.session_state.chat_sessions.values()
        if chat["dataset_id"] == dataset_id
    ]


def dataset_has_chat(dataset_id):
    return len(get_dataset_chats(dataset_id)) > 0


def create_chat_if_needed(dataset_id):
    """
    Creates the first chat for a dataset.
    """

    chats = get_dataset_chats(dataset_id)

    if chats:
        latest_chat = chats[-1]
        set_current_chat(latest_chat["id"])
        return latest_chat["id"]

    return create_chat(dataset_id)


# ==========================================================
# Convenience Helpers
# ==========================================================

def get_current_dataframe():
    dataset = get_current_dataset()

    if dataset is None:
        return None

    return dataset["dataframe"]


def get_current_retriever():
    dataset = get_current_dataset()

    if dataset is None:
        return None

    return dataset["retriever"]


def get_current_vectorstore():
    dataset = get_current_dataset()

    if dataset is None:
        return None

    return dataset["vectorstore"]