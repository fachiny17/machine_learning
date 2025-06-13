import streamlit as st
from typing import Generator
from groq import Groq
import uuid

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

# Page configuration
st.set_page_config(
    page_title="NucleiScan AI: Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

if st.button("← Back to Home"):
    st.session_state.current_page = None
    st.switch_page("../app/NucleiScan_AI.py")

st.title("NucleiScan AI: Chatbot")

# Initialize Firebase (only once)
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("../nucleiscan-ai-29e52debe285.json")  # Use actual filename
        firebase_admin.initialize_app(cred)
    except FirebaseError as e:
        st.error(f"Firebase initialization failed: {e}")
        st.stop()

# Initialize Firestore client
db = firestore.client()

# Initialize Groq client
try:
    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"],
    )
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# Model configuration
MODEL_CONFIG = {
    "name": "Compound-Beta",
    "tokens": 8192,
    "developer": "Groq"
}

# Firestore collection names
CHATS_COLLECTION = "nucleiscan_chats"

st.markdown("Chat with NucleiScan AI")  # Fixed typo in "NucleiScan"

# Load chat history from Firestore
def load_chat_history(user_id="default_user"):
    try:
        chats_ref = db.collection(CHATS_COLLECTION).where("user_id", "==", user_id)
        docs = chats_ref.stream()
        return {doc.id: doc.to_dict() for doc in docs}
    except FirebaseError as e:
        st.error(f"Error loading chat history: {e}")
        return {}

# Save chat history to Firestore
def save_chat_to_firestore(chat_id, chat_data, user_id="default_user"):
    try:
        doc_ref = db.collection(CHATS_COLLECTION).document(chat_id)
        doc_ref.set({
            **chat_data,
            "user_id": user_id,
            "title": chat_data.get("title", "New Chat"),
            "messages": chat_data.get("messages", []),
            "last_updated": firestore.SERVER_TIMESTAMP
        })
        return True
    except FirebaseError as e:
        st.error(f"Error saving chat history: {e}")
        return False

# Delete chat from Firestore
def delete_chat_from_firestore(chat_id):
    try:
        db.collection(CHATS_COLLECTION).document(chat_id).delete()
        return True
    except FirebaseError as e:
        st.error(f"Error deleting chat: {e}")
        return False

# Generate chat responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        else:
            yield ""

# Load chat history on initial load
if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history()

# Sidebar for chat history management
with st.sidebar:
    st.header("Chat History")

    # Button to start new chat
    if st.button("➕ New Chat"):
        st.session_state.active_chat = None
        st.session_state.current_chat = []
        st.session_state.messages = []
        st.rerun()

    # Clear all history button
    if st.button("🗑️ Clear All History", type="primary"):
        for chat_id in list(st.session_state.chat_history.keys()):
            if delete_chat_from_firestore(chat_id):
                del st.session_state.chat_history[chat_id]
        if st.session_state.active_chat:
            delete_chat_from_firestore(st.session_state.active_chat)
        st.session_state.active_chat = None
        st.session_state.current_chat = []
        st.session_state.messages = []
        st.rerun()

    # Display saved chats with delete buttons
    for chat_id, chat_data in list(st.session_state.chat_history.items()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"💬 {chat_data.get('title', 'Untitled Chat')[:30]}...",
                key=f"btn_{chat_id}",
                use_container_width=True
            ):
                st.session_state.active_chat = chat_id
                st.session_state.current_chat = chat_data.get("messages", [])
                st.rerun()
        with col2:
            if st.button(
                "❌", 
                key=f"del_{chat_id}",
                help="Delete this chat"
            ):
                if delete_chat_from_firestore(chat_id):
                    del st.session_state.chat_history[chat_id]
                    if st.session_state.active_chat == chat_id:
                        st.session_state.active_chat = None
                        st.session_state.current_chat = []
                    st.rerun()

# Main chat area
for message in st.session_state.current_chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to current chat
    st.session_state.current_chat.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    try:
        with st.spinner("Thinking..."):
            chat_completion = client.chat.completions.create(
                model=MODEL_CONFIG['name'].lower(),
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.current_chat
                ],
                max_tokens=MODEL_CONFIG['tokens'],
                stream=True
            )

            with st.chat_message("assistant"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)

                # Add AI response to current chat
                st.session_state.current_chat.append(
                    {"role": "assistant", "content": full_response}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

                # Save to Firestore
                if st.session_state.active_chat is None:
                    chat_id = str(uuid.uuid4())  # Generate unique ID
                    chat_data = {
                        "title": prompt[:50],  # First 50 chars as title
                        "messages": st.session_state.current_chat
                    }
                    if save_chat_to_firestore(chat_id, chat_data):
                        st.session_state.chat_history[chat_id] = chat_data
                        st.session_state.active_chat = chat_id
                        st.rerun()
                else:
                    # Update existing chat in Firestore
                    chat_data = {
                        "title": st.session_state.chat_history[st.session_state.active_chat].get("title", "New Chat"),
                        "messages": st.session_state.current_chat
                    }
                    if save_chat_to_firestore(st.session_state.active_chat, chat_data):
                        st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")