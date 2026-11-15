from functools import lru_cache
from finagent.config.llm import create_llm, invoke_llm

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from finagent.rag.pipeline import ask_rag

from finagent.market.pipeline import (
    analyze_market
)

from finagent.forecast.pipeline import (
    forecast_stock
)

from .state import FinAgentState


# ============================================================
# RAG AGENT
# ============================================================

def rag_node(
    state: FinAgentState
):

    answers, sources = [], []
    for company in state.get("companies") or [None]:
        result = ask_rag(state["question"], company=company)
        answers.append(f"{company or 'News'}:\n{result['answer']}")
        for i, doc in enumerate(result.get("documents", []), 1):
            meta = doc.metadata
            sources.append({
                "company": meta.get("company"), "label": f"Source {i}",
                "title": meta.get("article_title"), "date": str(meta.get("date", "")),
                "source": meta.get("source"), "url": meta.get("article_url"),
            })
    return {"rag_answer": "\n\n".join(answers), "sources": sources}


# ============================================================
# MARKET AGENT
# ============================================================

def market_node(
    state: FinAgentState
):

    answers = []
    for company in state.get("companies") or [None]:
        result = analyze_market(state["question"], company=company)
        answers.append(f"{company or 'Market'}:\n{result['answer']}")
    return {"market_answer": "\n\n".join(answers)}


# ============================================================
# FORECAST AGENT
# ============================================================

def forecast_node(
    state: FinAgentState
):

    answers = []
    for company in state.get("companies") or [None]:
        result = forecast_stock(state["question"], company=company)
        answers.append(f"{company or 'Forecast'}:\n{result['answer']}")
    return {"forecast_answer": "\n\n".join(answers)}


# ============================================================
# SYNTHESIS LLM
# ============================================================



# ============================================================
# SYNTHESIS PROMPT
# ============================================================

synthesis_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the synthesis agent of FinAgent-X.

Your task is to combine the analyses produced
by specialized financial agents into one clear,
coherent answer.

Possible sources:

- News/RAG Agent:
  explains financial news and company events.

- Market Agent:
  interprets market data and technical indicators.

- Forecast Agent:
  provides an experimental model-based estimate
  of the next-session return.

Rules:

1. Use ONLY the agent outputs provided.

2. Do not invent financial facts.

3. Do not invent missing agent results.

4. Clearly distinguish:
   - news/events,
   - technical market signals,
   - forecasting estimates.

5. Forecasts are uncertain quantitative signals,
   not guaranteed future outcomes.

6. Answer the user's original question directly.

7. Be concise but sufficiently explanatory.

8. History is only for resolving references and follow-up intent. Never reuse
   historical figures or previous agent outputs as fresh evidence.

9. For comparisons, clearly attribute each observation and news citation to its
   company. Include the company with citations, e.g. [Maybank, Source 1].
"""
            ),

            (
                "human",
                """
USER QUESTION:

{question}

CONVERSATION HISTORY (context only):
{history}


SELECTED AGENTS:

{selected_agents}


NEWS / RAG ANALYSIS:

{rag_answer}


MARKET ANALYSIS:

{market_answer}


FORECAST ANALYSIS:

{forecast_answer}


FINAL SYNTHESIS:
"""
            )
        ]
    )
)


@lru_cache(maxsize=1)
def get_synthesis_chain():
    return synthesis_prompt | create_llm() | StrOutputParser()


# ============================================================
# SYNTHESIS NODE
# ============================================================

def synthesis_node(
    state: FinAgentState
):

    final_answer = invoke_llm(get_synthesis_chain(),
        {
            "history": state.get("conversation_history", []),
            "question":
                state["question"],

            "selected_agents":
                ", ".join(
                    state.get(
                        "selected_agents",
                        []
                    )
                ),

            "rag_answer":
                state.get(
                    "rag_answer",
                    ""
                ),

            "market_answer":
                state.get(
                    "market_answer",
                    ""
                ),

            "forecast_answer":
                state.get(
                    "forecast_answer",
                    ""
                )
        }
    )

    return {
        "final_answer":
            final_answer
    }
