from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from finagent.config.universe import universe


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


FORECAST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocessed_data"
    / "forecasting_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@lru_cache(maxsize=1)
def load_forecast_data():

    data = pd.read_csv(
        FORECAST_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    return data


# ============================================================
# GET LATEST WINDOW
# ============================================================

def get_latest_window(
    company,
    feature_columns,
    window_size
):

    data = load_forecast_data()

    ticker = universe[
        company
    ]["ticker"]

    company_data = data[
        data["ticker"] == ticker
    ].copy()

    company_data = company_data.sort_values(
        "date"
    )


    # Sécurité : inf → NaN
    company_data[
        feature_columns
    ] = (
        company_data[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    # Garder uniquement les lignes avec toutes les features
    company_data = company_data.dropna(
        subset=feature_columns
    )


    if len(company_data) < window_size:
        return None


    latest_window = company_data.tail(
        window_size
    ).copy()


    return latest_window
