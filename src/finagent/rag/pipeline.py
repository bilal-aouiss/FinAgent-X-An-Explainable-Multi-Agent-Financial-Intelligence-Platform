from .retrieval import retrieve_with_policy
from .generation import build_context, get_generation_chain
from finagent.config.llm import invoke_llm


def ask_rag(question, k=5, *, company=None):
    result = retrieve_with_policy(question, k=k, fetch_k=max(20, k), company=company)
    if not result.documents:
        answer = result.notice
    else:
        answer = invoke_llm(get_generation_chain(), {
            "question": question,
            "context": build_context(result.documents),
        })
        if result.notice:
            answer = result.notice + "\n\n" + answer
    return {"question": question, "answer": answer, "documents": result.documents,
            "time_window": result.time_window, "retrieval_notice": result.notice}


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
