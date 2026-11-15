from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
import os
import json
import hashlib

import numpy as np

from finagent.config.company_matching import resolve_company, identification_message
from finagent.config.universe import universe
from .recency import DateWindow, eligible_provenance, parse_date, provenance_records, resolve_time_window

DOCUMENTS_PATH = Path(__file__).resolve().parents[2] / "documents.pkl"


@lru_cache(maxsize=1)
def load_documents():
    with DOCUMENTS_PATH.open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    # The product loads the saved index. It must NEVER rebuild it on a chat request.
    root = Path(__file__).resolve().parents[3]
    index_dir = Path(os.getenv("FINAGENT_FAISS_DIR", str(root / "models" / "rag")))
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["documents_sha256"] != hashlib.sha256(DOCUMENTS_PATH.read_bytes()).hexdigest():
        raise RuntimeError("The saved RAG index does not match documents.pkl.")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    # Only project-owned artifacts, never uploads or remote pickle files.
    return FAISS.load_local(str(index_dir), embedding, allow_dangerous_deserialization=True)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)


@dataclass
class RetrievalResult:
    documents: list
    time_window: DateWindow | None = None
    notice: str = ""


def retrieve_with_policy(query, k=5, fetch_k=20, *, corpus=None, store=None, reranker=None, company=None):
    if k < 1 or fetch_k < k:
        raise ValueError("Require 1 <= k <= fetch_k.")
    injected_corpus = corpus is not None
    corpus = load_documents() if corpus is None else corpus
    if not corpus:
        return RetrievalResult([], notice="No news corpus is available.")
    latest = max(parse_date(p["date"]) for doc in corpus for p in provenance_records(doc.metadata))
    try:
        window = resolve_time_window(query, latest)
    except ValueError as exc:
        return RetrievalResult([], notice=str(exc))
    resolution = resolve_company(query)
    if company is not None and company not in universe:
        raise ValueError("Unknown company scope.")
    if company is None and resolution.status in {"ambiguous", "outside_universe"}:
        return RetrievalResult([], window, identification_message(resolution))
    company = company or resolution.company

    def eligible(metadata):
        return (company is None or metadata.get("company") == company) and bool(eligible_provenance(metadata, window))

    available = [doc for doc in corpus if company is None or doc.metadata.get("company") == company]
    eligible_docs = [doc for doc in available if eligible(doc.metadata)]
    if not eligible_docs:
        scope = f" for {company}" if company else ""
        notice = f"No news is available{scope} in the requested publication window."
        if window:
            notice += " " + window.describe()
        if available:
            last_company_date = max(parse_date(p["date"]) for doc in available for p in provenance_records(doc.metadata))
            notice += f" Latest available publication{scope}: {last_company_date}."
        return RetrievalResult([], window, notice)
    if store is None:
        if injected_corpus:
            raise ValueError("An injected corpus requires its matching vector store.")
        store = get_vectorstore()
    kwargs = {"k": fetch_k}
    if window is not None or company is not None:
        # FAISS applies metadata filtering AFTER its vector search. Search all
        # indexed rows before filtering so eligible docs beyond global Top-20
        # cannot be silently lost. The reranker still receives at most fetch_k.
        kwargs.update(filter=eligible, fetch_k=len(corpus))
    candidates = [doc for doc in store.similarity_search(query, **kwargs) if eligible(doc.metadata)][:fetch_k]
    if not candidates:
        return RetrievalResult([], window, "No matching news was retrieved within the requested scope.")
    reranker = get_reranker() if reranker is None else reranker
    scores = reranker.predict([[query, doc.page_content] for doc in candidates])
    chosen = [candidates[i] for i in np.argsort(scores)[::-1][:k]]
    # A deduplicated document may have several publication dates. Display an
    # origin inside the requested window, never the older canonical row's date.
    if window is not None:
        scoped = []
        for doc in chosen:
            origin = max(eligible_provenance(doc.metadata, window), key=lambda p: parse_date(p["date"]))
            scoped_doc = copy(doc)
            scoped_doc.metadata = {**doc.metadata,
                **{key: origin[key] for key in ("date", "article_title", "article_url", "nid", "info_index") if key in origin},
                "matched_provenance_doc_id": origin.get("doc_id", doc.metadata.get("doc_id"))}
            scoped.append(scoped_doc)
        chosen = scoped
    return RetrievalResult(chosen, window, window.describe() if window else "")


def retrieve_and_rerank(query, k=5, fetch_k=20):
    """Keep the existing list-returning API; ask_rag also exposes policy notices."""
    return retrieve_with_policy(query, k=k, fetch_k=fetch_k).documents
