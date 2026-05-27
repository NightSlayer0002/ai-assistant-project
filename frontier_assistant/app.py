"""Streamlit chat interface for the Frontier (Gemini) assistant."""

import streamlit as st
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frontier_assistant.assistant import FrontierAssistant
from observability.logger import SimpleLogger

st.set_page_config(page_title="Frontier Assistant - Gemini", page_icon="🧠", layout="centered")
st.title("🧠 Frontier Assistant (Gemini)")
st.caption("Powered by Google Gemini 2.5 Flash")

with st.sidebar:
    st.header("Settings")
    st.info("Model: Gemini 2.5 Flash")
    st.markdown("**Tools Available:** Time, Calculator")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.assistant = FrontierAssistant()
        st.session_state.messages = []
        st.rerun()

if "assistant" not in st.session_state:
    st.session_state.assistant = FrontierAssistant()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logger" not in st.session_state:
    st.session_state.logger = SimpleLogger()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            response = st.session_state.assistant.send_message(user_input)
            latency_ms = (time.time() - start_time) * 1000

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.session_state.logger.log_interaction(
            model_name="gemini-2.5-flash",
            prompt=user_input,
            response=response,
            latency_ms=latency_ms
        )
