"""Retrieval-only deduplication. Source rows and forecasting news stay intact."""

import hashlib
import json
import pandas as pd

PROVENANCE_COLUMNS = ("doc_id", "nid", "info_index", "date", "article_url", "article_title")


def deduplicate_rag_rows(data: pd.DataFrame) -> pd.DataFrame:
    """One canonical row per company/ticker/text_hash, with every origin retained.

    Canonical row = earliest date, then lexicographically smallest doc_id.
    Hashes are checked against text_clean so corrupt hashes cannot merge texts.
    The returned provenance is a JSON-compatible list of records, not a CSV edit.
    """
    required = {"company", "ticker", "text_hash", "text_clean", *PROVENANCE_COLUMNS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing RAG columns: {sorted(missing)}")
    if data[["company", "ticker", "text_hash", "text_clean", "doc_id", "date"]].isna().any().any():
        raise ValueError("RAG identity, text and publication date must not be missing.")
    expected = data["text_clean"].map(lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest())
    if not expected.eq(data["text_hash"]).all():
        raise ValueError("text_hash does not match text_clean.")
    rows = []
    ordered = data.sort_values(["date", "doc_id"], kind="stable")
    for _, group in ordered.groupby(["company", "ticker", "text_hash"], sort=False):
        row = group.iloc[0].copy()
        # JSON round-trip converts numpy scalar values to portable Python values.
        row["provenance"] = json.loads(group[list(PROVENANCE_COLUMNS)].to_json(orient="records", date_format="iso"))
        row["original_doc_ids"] = group["doc_id"].astype(str).tolist()
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else data.assign(provenance=[], original_doc_ids=[])
