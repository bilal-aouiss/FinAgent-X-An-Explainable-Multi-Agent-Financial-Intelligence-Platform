from pathlib import Path
from functools import lru_cache
from finagent.config.universe import universe
from finagent.config.company_matching import identify_company
import pandas as pd

# ============================================================
# 1. DATA PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MARKET_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed_data"
    / "malaysia_market_features.csv"
)




# ============================================================
# 2. LOAD MARKET DATA
# ============================================================

@lru_cache(maxsize=1)
def load_market_data():

    data = pd.read_csv(
        MARKET_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    return data


# ============================================================
# 3. GET LATEST MARKET OBSERVATION
# ============================================================

def get_latest_market_data(
    company
):

    data = load_market_data()

    ticker = universe[
        company
    ]["ticker"]

    company_data = data[
        data["ticker"] == ticker
    ].copy()

    if company_data.empty:

        return None

    company_data = company_data.sort_values("date")

    latest = company_data.iloc[-1]

    return latest
