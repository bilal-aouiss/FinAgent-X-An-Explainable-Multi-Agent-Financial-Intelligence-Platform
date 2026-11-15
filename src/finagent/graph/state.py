from typing import TypedDict


class FinAgentState(TypedDict, total=False):

    question: str
    conversation_history: list[dict[str, str]]
    companies: list[str]
    sources: list[dict]

    selected_agents: list[str]

    rag_answer: str
    market_answer: str
    forecast_answer: str

    final_answer: str
