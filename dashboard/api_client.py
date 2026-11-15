"""The dashboard talks only HTTP; ML models live in the backend process."""

import os
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.getenv("FINAGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")


class APIError(Exception):
    pass


def request(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=(5, 300), **kwargs)
    except requests.Timeout as exc:
        raise APIError("The analysis timed out. Your previous conversation is still available; check the backend before retrying.") from exc
    except requests.RequestException as exc:
        raise APIError("Cannot reach FinAgent-X. Start the FastAPI backend and refresh this page.") from exc
    if not response.ok:
        try:
            detail = response.json().get("detail", "The backend could not complete this request.")
            if not isinstance(detail, str):
                detail = "Please check the message and company selection."
        except ValueError:
            detail = "The backend could not complete this request."
        raise APIError(detail)
    return response.json()


@st.cache_data(ttl=120, show_spinner=False)
def get(path):
    return request("GET", path)


def company_path(endpoint, company):
    return f"/{endpoint}/{quote(company, safe='')}"


def chat(message, session_id):
    return request("POST", "/chat", json={"message": message, "session_id": session_id})
