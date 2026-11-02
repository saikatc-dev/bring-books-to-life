"""Streamlit chat UI for Bring Books to Life.

Run:  streamlit run app.py
Needs GROQ_API_KEY in the environment / .env (free Groq key).
"""

import hashlib

import streamlit as st

from agent import BookChat
from personas import PERSONAS, DEFAULT

st.set_page_config(page_title="Bring Books to Life", page_icon="🎭", layout="centered")
st.title("🎭 Bring Books to Life")
st.caption(
    "Chat with a character from a book. They stay in character, answer from the text, and remember you."
)

with st.sidebar:
    key = st.selectbox(
        "Character",
        list(PERSONAS),
        format_func=lambda k: PERSONAS[k].name,
        index=list(PERSONAS).index(DEFAULT),
    )
    uploaded = st.file_uploader(
        "Ground this character in your own source (.txt) — keeps their persona",
        type=["txt"],
    )
    uploaded_bytes = uploaded.getvalue() if uploaded else None
    if uploaded is not None and not uploaded_bytes:
        st.warning("The uploaded text file is empty; using the character's default source.")
    if uploaded_bytes:
        try:
            src = uploaded_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("The uploaded file must be a UTF-8 encoded text file.")
            st.stop()
        source_id = hashlib.sha256(uploaded_bytes).hexdigest()
        st.caption(f"RAG source: uploaded file — {uploaded.name}")
    else:
        src = None
        source_id = None
        st.caption(f"RAG source: {PERSONAS[key].source_file}")
    reset_requested = st.button("Start / reset conversation", type="primary")

switch_requested = (
    "chat" not in st.session_state or st.session_state.chat.persona.key != key
)
source_changed = st.session_state.get("source_id") != source_id
if reset_requested or switch_requested or source_changed:
    st.session_state.chat = BookChat(key, source_text=src)
    st.session_state.history = []
    st.session_state.source_id = source_id

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if prompt := st.chat_input(f"Say something to {st.session_state.chat.persona.name}…"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("thinking in character…"):
            reply = st.session_state.chat.say(prompt)
        st.write(reply)
    st.session_state.history.append(("assistant", reply))
