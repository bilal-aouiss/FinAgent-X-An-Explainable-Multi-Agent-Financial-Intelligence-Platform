from typing import Literal
from functools import lru_cache

from pydantic import BaseModel, Field

from finagent.config.llm import create_llm, invoke_llm

from .state import FinAgentState


# ============================================================
# ROUTER OUTPUT
# ============================================================

class RouteDecision(BaseModel):

    agents: list[
        Literal[
            "rag",
            "market",
            "forecast"
        ]
    ] = Field(
        min_length=1
    )


# ============================================================
# ROUTER LLM
# ============================================================

@lru_cache(maxsize=1)
def get_structured_router():
    return create_llm().with_structured_output(RouteDecision)


# ============================================================
# ROUTER PROMPT
# ============================================================

router_prompt = """
You are the router of FinAgent-X,
a multi-agent financial intelligence system
focused on Malaysian listed companies.

Select ONE OR MORE agents required
to answer the user's question.

Available agents:

rag:
Use for financial news, company events,
announcements, earnings reported in news,
management changes, acquisitions, contracts,
and explanations based on The Edge Malaysia.

market:
Use for current or historical market data
and technical indicators such as price,
returns, volume, RSI, MACD, moving averages,
volatility, momentum, ATR or Bollinger Bands.

forecast:
Use when the user asks about future price
direction, expected return, prediction,
forecast or outlook based on the forecasting model.


Examples:

"What did The Edge report about Maybank?"
→ ["rag"]

"What is Maybank's RSI?"
→ ["market"]

"Predict Maybank's next-day return."
→ ["forecast"]

"Explain Maybank's recent news and technical situation."
→ ["rag", "market"]

"Analyze Maybank using news, technical indicators
and the forecast."
→ ["rag", "market", "forecast"]


Select only the agents genuinely needed.
The history is for resolving follow-up intent only. Select agents for the CURRENT
question; do not repeat previous requests just because they appear in history.

CONVERSATION HISTORY:
{history}

QUESTION:

{question}
"""


# ============================================================
# ROUTER NODE
# ============================================================

def router_node(
    state: FinAgentState
):

    prompt = router_prompt.format(
        question=state["question"],
        history=state.get("conversation_history", []),
    )

    decision = invoke_llm(get_structured_router(), prompt)

    agents = list(
        dict.fromkeys(
            decision.agents
        )
    )

    return {
        "selected_agents":
            agents
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def choose_agents(
    state: FinAgentState
):

    mapping = {
        "rag":
            "rag_agent",

        "market":
            "market_agent",

        "forecast":
            "forecast_agent"
    }

    return [
        mapping[agent]
        for agent
        in state["selected_agents"]
    ]
