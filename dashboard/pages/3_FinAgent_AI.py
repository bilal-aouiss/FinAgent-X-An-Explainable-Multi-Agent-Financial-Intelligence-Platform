import uuid

import streamlit as st

from api_client import APIError, chat
from ui import show_agent_tags, show_sources

st.title("FinAgent AI")
st.caption("Ask about news, technical indicators or experimental forecasts across 51 Malaysian companies.")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

left, right = st.columns([4, 1])
with left:
    st.caption("Follow-ups retain company context within this conversation.")
with right:
    if st.button("New conversation", width="stretch"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("**Start with a company and a question**")
        st.write("Analyze Maybank using recent news and technical indicators.")
        st.write("What about its RSI?")
        st.write("Compare it with CIMB.")

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant":
            show_agent_tags(entry.get("selected_agents", []))
            show_sources(entry.get("sources", []))

message = st.chat_input("Ask FinAgent-X…", max_chars=4000)
if message and message.strip():
    with st.chat_message("user"):
        st.markdown(message)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Routing your question and preparing the analysis…"):
                result = chat(message, st.session_state.session_id)
            st.markdown(result["answer"])
            show_agent_tags(result["selected_agents"])
            show_sources(result.get("sources", []))
        st.session_state.messages.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"], "selected_agents": result["selected_agents"], "sources": result.get("sources", [])},
        ])
    except APIError as exc:
        st.error(str(exc))
        st.caption("The response could not be displayed. Your earlier messages are preserved.")

st.caption("Experimental model-based quantitative signals. Not investment advice.")
