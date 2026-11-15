import os
from pathlib import Path
from threading import Lock
from time import monotonic, sleep

from dotenv import load_dotenv

_lock = Lock()
_last_call = 0.0


def create_llm():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Configure GOOGLE_API_KEY in src/.env to enable FinAgent AI.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, timeout=60, max_retries=1,
    )


def invoke_llm(chain, inputs):
    global _last_call
    with _lock:
        interval = max(0.0, float(os.getenv("FINAGENT_LLM_INTERVAL_SECONDS", "6")))
        delay = interval - (monotonic() - _last_call)
        if delay > 0:
            sleep(delay)
        try:
            return chain.invoke(inputs)
        finally:
            _last_call = monotonic()
