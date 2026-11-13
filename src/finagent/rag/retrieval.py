import pickle
from pathlib import Path

import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


# ============================================================
# 1. CHARGEMENT DES DOCUMENTS
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[2]

DOCUMENTS_PATH = SRC_DIR / "documents.pkl"


with open(DOCUMENTS_PATH, "rb") as f:
    documents = pickle.load(f)


print(
    "Nombre de documents :",
    len(documents)
)


# ============================================================
# 2. MODELE D'EMBEDDINGS
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ============================================================
# 3. INDEX FAISS
# ============================================================

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)


print(
    "Index FAISS créé."
)


# ============================================================
# 4. CROSS-ENCODER RERANKER
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# 5. RETRIEVAL + RERANKING
# ============================================================

def retrieve_and_rerank(
    query,
    k=5,
    fetch_k=20
):
    """
    1. FAISS récupère fetch_k candidats
    2. Le CrossEncoder les rerank
    3. On conserve les k meilleurs
    """

    # 1. Candidate retrieval avec FAISS
    candidates = vectorstore.similarity_search(
        query,
        k=fetch_k
    )


    # 2. Construire les couples :
    # [question, document]
    pairs = [
        [
            query,
            doc.page_content
        ]
        for doc in candidates
    ]


    # 3. Scores du CrossEncoder
    scores = reranker.predict(
        pairs
    )


    # 4. Trier du score le plus élevé au plus faible
    ranked_indices = np.argsort(
        scores
    )[::-1]


    # 5. Top k final
    reranked_docs = [
        candidates[i]
        for i in ranked_indices[:k]
    ]


    return reranked_docs