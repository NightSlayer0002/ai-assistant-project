"""Streamlit web interface for the OSS Assistant chatbot."""

import streamlit as st
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oss_assistant.assistant import OSSAssistant
from oss_assistant.guardrails import check_input, check_output
from observability.logger import SimpleLogger

# set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="OSS Assistant - Qwen2.5",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 OSS Assistant (Qwen2.5)")
st.caption("Powered by HuggingFace Inference API • Open Source Model")

with st.sidebar:
    st.header("Settings")
    st.info("Model: Qwen2.5-7B-Instruct")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.assistant = OSSAssistant()
        st.session_state.messages = []
        st.rerun()

# Session state persists data across Streamlit reruns
if "assistant" not in st.session_state:
    st.session_state.assistant = OSSAssistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "logger" not in st.session_state:
    st.session_state.logger = SimpleLogger()

# Redraw chat history on each rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):

    is_safe, reason = check_input(user_input)

    if not is_safe:
        st.warning(f"⚠️ {reason}")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = st.session_state.assistant.send_message(user_input)
                latency_ms = (time.time() - start_time) * 1000

            output_safe, output_reason = check_output(response)
            if not output_safe:
                st.warning(f"⚠️ Output flagged: {output_reason}")

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            st.session_state.logger.log_interaction(
                model_name="Qwen2.5-7B-Instruct",
                prompt=user_input,
                response=response,
                latency_ms=latency_ms
            )
