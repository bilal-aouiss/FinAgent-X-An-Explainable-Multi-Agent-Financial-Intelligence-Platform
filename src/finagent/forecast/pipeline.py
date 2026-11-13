from pathlib import Path

import joblib
import numpy as np

from tensorflow.keras.models import load_model

from finagent.market.data import identify_company
from .data import get_latest_window


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecast"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

model = load_model(
    MODEL_DIR
    / "lstm_forecast.keras"
)


scaler = joblib.load(
    MODEL_DIR
    / "feature_scaler.pkl"
)


feature_columns = joblib.load(
    MODEL_DIR
    / "feature_columns.pkl"
)


window_size = joblib.load(
    MODEL_DIR
    / "window_size.pkl"
)


# ============================================================
# FORECAST PIPELINE
# ============================================================

def forecast_stock(
    question
):

    # --------------------------------------------------------
    # 1. Identifier l'entreprise
    # --------------------------------------------------------

    company = identify_company(
        question
    )


    if company is None:

        return {
            "company": None,
            "predicted_return": None,
            "direction": None,
            "answer": (
                "I could not identify a company "
                "from the FinAgent-X universe."
            )
        }


    # --------------------------------------------------------
    # 2. Récupérer les 60 derniers jours
    # --------------------------------------------------------

    window = get_latest_window(
        company=company,
        feature_columns=feature_columns,
        window_size=window_size
    )


    if window is None:

        return {
            "company": company,
            "predicted_return": None,
            "direction": None,
            "answer": (
                f"Not enough historical data "
                f"is available for {company}."
            )
        }


    # --------------------------------------------------------
    # 3. Scaling
    # --------------------------------------------------------

    X = scaler.transform(
        window[
            feature_columns
        ]
    )


    # LSTM attend :
    # (samples, time_steps, features)

    X = np.expand_dims(
        X,
        axis=0
    )


    # --------------------------------------------------------
    # 4. Prediction
    # --------------------------------------------------------

    predicted_return = float(
        model.predict(
            X,
            verbose=0
        )[0][0]
    )


    # --------------------------------------------------------
    # 5. Direction
    # --------------------------------------------------------

    if predicted_return > 0:

        direction = "UP"

    elif predicted_return < 0:

        direction = "DOWN"

    else:

        direction = "FLAT"


    latest_date = (
        window["date"]
        .iloc[-1]
        .date()
    )


    # --------------------------------------------------------
    # 6. Réponse
    # --------------------------------------------------------

    answer = (
        f"Based on the latest {window_size} trading sessions "
        f"available through {latest_date}, the LSTM forecasting "
        f"model estimates {company}'s next-session return at "
        f"{predicted_return:.2f}%. "
        f"The estimated direction is {direction}. "
        f"This is a model-based quantitative signal, "
        f"not a guaranteed market outcome."
    )


    return {
        "company": company,
        "date": str(latest_date),
        "predicted_return": predicted_return,
        "direction": direction,
        "answer": answer
    }