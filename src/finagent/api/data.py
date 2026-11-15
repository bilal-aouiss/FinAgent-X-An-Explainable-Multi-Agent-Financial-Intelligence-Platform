"""Read-only views for the dashboard; indicator and prediction logic stays in agents."""

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from finagent.config.universe import universe
from finagent.market.data import load_market_data, get_latest_market_data
from finagent.forecast.pipeline import forecast_stock

ROOT = Path(__file__).resolve().parents[3]
_forecast_lock = Lock()


def records(frame):
    """Pandas/numpy dates and missing or infinite values become valid JSON."""
    frame = frame.replace([np.inf, -np.inf], np.nan).copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def market_snapshot(company):
    row = get_latest_market_data(company)
    if row is None:
        return None
    return records(pd.DataFrame([row]))[0]


def market_overview():
    data = load_market_data()
    data = data[data["ticker"].isin([v["ticker"] for v in universe.values()])]
    latest = data.sort_values("date").groupby("ticker", sort=False).tail(1)
    return records(latest.sort_values("company"))


def price_history(company, period):
    data = load_market_data()
    frame = data[data["ticker"] == universe[company]["ticker"]].sort_values("date")
    if not frame.empty and period != "MAX":
        months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[period]
        frame = frame[frame["date"] >= frame["date"].max() - pd.DateOffset(months=months)]
    columns = [c for c in [
        "date", "open", "high", "low", "close", "volume",
        "sma_20", "sma_50", "sma_200", "rsi_14",
        "macd", "macd_signal", "macd_histogram",
        "bollinger_upper", "bollinger_middle", "bollinger_lower",
    ] if c in frame]
    return records(frame[columns])


@lru_cache(maxsize=1)
def news_data():
    return pd.read_csv(ROOT / "data/preprocessed_data/news_data_sentiment.csv")


def company_news(company, limit):
    data = news_data()
    frame = data[data["ticker"] == universe[company]["ticker"]].sort_values("date", ascending=False)
    frame = frame.drop_duplicates(["article_url", "text"]).head(limit)
    result = []
    for _, row in frame.iterrows():
        url = row.get("article_url")
        result.append({
            "title": row.get("article_title"), "date": row.get("date"),
            "sentiment": row.get("sentiment"), "text": row.get("text"),
            "url": url, "source": urlparse(str(url)).hostname,
        })
    return json.loads(pd.DataFrame(result).to_json(orient="records")) if result else []


def forecast(company):
    with _forecast_lock:
        result = forecast_stock(company, company=company)
    if result.get("predicted_return") is None:
        raise RuntimeError(result["answer"])
    return {
        "company": company, "ticker": universe[company]["ticker"],
        "prediction_date": datetime.now(timezone.utc).date().isoformat(),
        "last_observation_date": result["date"],
        "predicted_return": result["predicted_return"], "direction": result["direction"],
    }


@lru_cache(maxsize=1)
def insights():
    return json.loads(Path(__file__).with_name("insights.json").read_text(encoding="utf-8"))
