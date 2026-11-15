from urllib.parse import urlparse

import streamlit as st

from api_client import APIError, get


def fetch(path):
    try:
        return get(path)
    except APIError as exc:
        st.error(str(exc))
        st.stop()


def number(value, digits=2, suffix=""):
    if value is None:
        return "Unavailable"
    return f"{value:,.{digits}f}{suffix}"


def source_link(url, label="Read source"):
    if isinstance(url, str) and urlparse(url).scheme in {"http", "https"}:
        st.link_button(label, url)


def show_sources(sources):
    if sources:
        with st.expander(f"Sources · {len(sources)}"):
            for source in sources:
                st.markdown(f"**{source.get('company') or 'News'} · {source.get('label') or 'Source'}**")
                st.write(source.get("title") or source.get("source") or "Financial news")
                st.caption(f"{source.get('date') or 'Date unavailable'} · {source.get('source') or ''}")
                source_link(source.get("url"))


def show_agent_tags(agents):
    labels = {"rag": "RAG", "market": "Market", "forecast": "Forecast"}
    if agents:
        st.caption("Agents used: " + " · ".join(labels[a] for a in agents if a in labels))
