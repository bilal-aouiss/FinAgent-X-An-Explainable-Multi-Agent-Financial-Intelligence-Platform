"""Small, side-effect-free market feature rules shared with the notebook."""

import numpy as np
import pandas as pd


def compute_volume_change(data: pd.DataFrame) -> pd.Series:
    """Percent change per ticker; a zero denominator is undefined (NaN)."""
    if not data.index.is_unique:
        raise ValueError("Volume change requires a unique row index.")
    ordered = data.sort_values(["ticker", "date"])
    previous = ordered.groupby("ticker")["volume"].shift(1).replace(0, np.nan)
    return ((ordered["volume"] / previous - 1) * 100).reindex(data.index)


def flag_extreme_returns(data: pd.DataFrame, threshold: float = 35) -> pd.DataFrame:
    """Inspection queue only: never delete or auto-adjust an extreme return."""
    return data.loc[data["daily_return"].abs() > threshold,
                    ["ticker", "date", "close", "daily_return"]].copy()
