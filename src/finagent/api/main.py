"""Run: uvicorn finagent.api.main:app --reload --app-dir src."""

from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from finagent.config.company_matching import resolve_company, identification_message
from finagent.config.universe import universe
from finagent.graph.graph import graph
from . import data
from .memory import get_session
from .schemas import ChatRequest, ChatResponse, Company, ForecastResponse

app = FastAPI(title="FinAgent-X", version="1.0.0", description="51-company Malaysian financial intelligence prototype.")


def canonical_company(value):
    result = resolve_company(value)
    if result.company is None:
        raise HTTPException(status_code=404, detail=identification_message(result))
    return result.company


@app.exception_handler(Exception)
async def service_error(request: Request, exc: Exception):
    # Do not return provider exceptions, credentials, or filesystem paths to clients.
    text = str(exc).lower()
    if any(term in text for term in ("429", "quota", "resource_exhausted", "rate limit")):
        detail = "Gemini quota or rate limit reached. Wait or restore the project quota before retrying. Market data and LSTM Forecast remain available."
        code = 429
    elif any(term in text for term in ("google_api_key", "api key", "api_key", "401", "403")):
        detail = "Gemini credentials are unavailable or invalid. Configure GOOGLE_API_KEY in src/.env and restart the API."
        code = 503
    elif isinstance(exc, (FileNotFoundError, ImportError)) or "saved rag index" in text:
        detail = "A required local artifact or dependency is unavailable. Check the backend configuration."
        code = 503
    elif "capacity reached" in text:
        detail = "RAM conversation capacity reached. Restart the API to clear sessions."
        code = 503
    else:
        detail = "The requested analysis is temporarily unavailable. Check the backend console before retrying."
        code = 503
    return JSONResponse(status_code=code, content={"detail": detail})


@app.get("/health")
def health():
    return {"status": "ok", "service": "FinAgent-X"}


@app.get("/companies", response_model=list[Company])
def companies():
    return [{"company": name, "ticker": info["ticker"]} for name, info in universe.items()]


@app.get("/market")
def overview():
    return data.market_overview()


@app.get("/market/{company}")
def market(company: str):
    result = data.market_snapshot(canonical_company(company))
    if result is None:
        raise HTTPException(404, "No market observations available for this company.")
    return result


@app.get("/company/{company}/history")
def history(company: str, period: Literal["1M", "3M", "6M", "1Y", "MAX"] = "6M"):
    return data.price_history(canonical_company(company), period)


@app.get("/company/{company}/news")
def news(company: str, limit: int = Query(default=8, ge=1, le=30)):
    return data.company_news(canonical_company(company), limit)


@app.get("/forecast/{company}", response_model=ForecastResponse)
def forecast(company: str):
    return data.forecast(canonical_company(company))


@app.get("/insights")
def insights():
    return data.insights()


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    session = get_session(body.session_id)
    # One turn at a time per session; other sessions keep independent histories.
    with session.lock:
        state = session.fresh_state(body.message)
        if len(state["companies"]) > 2:
            raise HTTPException(422, "Please compare at most two companies in one message.")
        # No checkpoint/thread_id and no old outputs passed to the graph.
        result = graph.invoke(state)
        answer = result.get("final_answer", "").strip()
        if not answer:
            raise RuntimeError("Empty graph synthesis.")
        session.remember(body.message, answer, state["companies"])
        return ChatResponse(
            session_id=body.session_id, answer=answer,
            selected_agents=result.get("selected_agents", []), sources=result.get("sources", []),
        )
