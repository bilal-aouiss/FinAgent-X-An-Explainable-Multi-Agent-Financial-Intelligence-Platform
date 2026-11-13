from .retrieval import retrieve_and_rerank
from .generation import build_context, generation_chain


# ============================================================
# 1. RAG COMPLET
# ============================================================

def ask_rag(
    question,
    k=5
):

    # ------------------------------------------
    # 1. Retrieval + reranking
    # ------------------------------------------

    documents = retrieve_and_rerank(
        query=question,
        k=k,
        fetch_k=20
    )


    # ------------------------------------------
    # 2. Construire le contexte
    # ------------------------------------------

    context = build_context(
        documents
    )


    # ------------------------------------------
    # 3. Génération
    # ------------------------------------------

    answer = generation_chain.invoke(
        {
            "question": question,
            "context": context
        }
    )


    return {
        "question": question,
        "answer": answer,
        "documents": documents
    }


# ============================================================
# 2. DISPLAY SOURCES
# ============================================================

def display_sources(
    documents
):

    print(
        "\nSOURCES:"
    )


    for i, doc in enumerate(
        documents,
        start=1
    ):

        print(
            f"\n[Source {i}]"
        )

        print(
            "Company:",
            doc.metadata["company"]
        )

        print(
            "Date:",
            doc.metadata["date"]
        )

        print(
            "URL:",
            doc.metadata["article_url"]
        )


# ============================================================
# 3. USER-FRIENDLY FUNCTION
# ============================================================

def ask_finagent(
    question
):

    result = ask_rag(
        question
    )


    print(
        "\nQUESTION\n"
    )

    print(
        result["question"]
    )


    print(
        "\nANSWER\n"
    )

    print(
        result["answer"]
    )


    print(
        "\n" + "=" * 100
    )


    display_sources(
        result["documents"]
    )