from langgraph.graph import (
    StateGraph,
    START,
    END
)

from .state import FinAgentState

from .router import (
    router_node,
    choose_agents
)

from .nodes import (
    rag_node,
    market_node,
    forecast_node,
    synthesis_node
)


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(
    FinAgentState
)


# ============================================================
# NODES
# ============================================================

builder.add_node(
    "router",
    router_node
)

builder.add_node(
    "rag_agent",
    rag_node
)

builder.add_node(
    "market_agent",
    market_node
)

builder.add_node(
    "forecast_agent",
    forecast_node
)

builder.add_node(
    "synthesis",
    synthesis_node
)


# ============================================================
# START → ROUTER
# ============================================================

builder.add_edge(
    START,
    "router"
)


# ============================================================
# ROUTER → ONE OR MULTIPLE AGENTS
# ============================================================

builder.add_conditional_edges(
    "router",
    choose_agents
)


# ============================================================
# SPECIALIZED AGENTS → SYNTHESIS
# ============================================================

builder.add_edge(
    "rag_agent",
    "synthesis"
)

builder.add_edge(
    "market_agent",
    "synthesis"
)

builder.add_edge(
    "forecast_agent",
    "synthesis"
)


# ============================================================
# SYNTHESIS → END
# ============================================================

builder.add_edge(
    "synthesis",
    END
)


# ============================================================
# COMPILE
# ============================================================

graph = builder.compile()