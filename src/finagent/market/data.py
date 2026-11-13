from pathlib import Path

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
# 2. FINAGENT UNIVERSE
# ============================================================

universe = {
    "Maybank": {
        "ticker": "1155.KL",
        "aliases": [
            "Maybank",
            "Malayan Banking"
        ]
    },

    "CIMB": {
        "ticker": "1023.KL",
        "aliases": [
            "CIMB",
            "CIMB Group"
        ]
    },

    "Public Bank": {
        "ticker": "1295.KL",
        "aliases": [
            "Public Bank"
        ]
    },

    "Tenaga Nasional": {
        "ticker": "5347.KL",
        "aliases": [
            "Tenaga Nasional",
            "TNB"
        ]
    },

    "Petronas Chemicals": {
        "ticker": "5183.KL",
        "aliases": [
            "Petronas Chemicals",
            "PETRONAS Chemicals",
            "PCG"
        ]
    }
}


# ============================================================
# 3. LOAD MARKET DATA
# ============================================================

def load_market_data():

    data = pd.read_csv(
        MARKET_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    return data


# ============================================================
# 4. IDENTIFY COMPANY FROM USER QUESTION
# ============================================================

def identify_company(question):

    question_lower = question.lower()

    for company, info in universe.items():

        # Company name
        if company.lower() in question_lower:
            return company

        if info["ticker"].lower() in question_lower:
            return company

        for alias in info["aliases"]:

            if alias.lower() in question_lower:
                return company

    return None


# ============================================================
# 5. GET LATEST MARKET OBSERVATION
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