import re
import pandas as pd

MARKET_TIMEZONE = "Asia/Kuala_Lumpur"
SIGNAL_HOUR = 18


def publication_timestamp(row) -> pd.Timestamp:
    """Prefer original timezone-aware timestamps; never guess time from `date`."""
    for column in ("datetime_my", "datetime"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value).strip()):
                raise ValueError("News publication time missing: date alone is insufficient.")
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                if column == "datetime":
                    raise ValueError("Naive datetime: the source timezone must be explicit.")
                timestamp = timestamp.tz_localize(MARKET_TIMEZONE)
            return timestamp.tz_convert(MARKET_TIMEZONE)
    if pd.notna(row.get("created")):
        return pd.to_datetime(row["created"], unit="ms", utc=True).tz_convert(MARKET_TIMEZONE)
    raise ValueError("News publication time missing: date alone is insufficient.")


def align_news_to_signals(news: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    result = news.copy()
    calendars = {}
    for ticker, group in market.groupby("ticker"):
        dates = pd.DatetimeIndex(pd.to_datetime(group["date"]))
        if dates.tz is not None:
            dates = dates.tz_convert(MARKET_TIMEZONE).tz_localize(None)
        dates = dates.normalize().unique().sort_values()
        calendars[ticker] = dates.tz_localize(MARKET_TIMEZONE) + pd.Timedelta(hours=SIGNAL_HOUR)
    publications, mapped, signal_times, statuses = [], [], [], []
    for _, row in result.iterrows():
        published = publication_timestamp(row)
        publications.append(published)
        signals = calendars.get(row["ticker"])
        status = "mapped"
        signal = pd.NaT
        if signals is None or len(signals) == 0:
            status = "unknown_ticker"
        elif published < signals[0].normalize():
            status = "before_market_history"
        else:
            position = signals.searchsorted(published, side="left")
            if position == len(signals):
                status = "after_last_signal"
            else:
                signal = signals[position]
        signal_times.append(signal)
        mapped.append(signal.tz_localize(None).normalize() if pd.notna(signal) else pd.NaT)
        statuses.append(status)
    result["publication_time_my"] = pd.Series(publications, index=result.index, dtype=f"datetime64[ns, {MARKET_TIMEZONE}]")
    result["market_date"] = pd.to_datetime(mapped)
    result["signal_time_my"] = pd.Series(signal_times, index=result.index, dtype=f"datetime64[ns, {MARKET_TIMEZONE}]")
    result["mapping_status"] = statuses
    valid = result["signal_time_my"].notna()
    if (result.loc[valid, "publication_time_my"] > result.loc[valid, "signal_time_my"]).any():
        raise AssertionError("A signal cannot use news that has not been published yet.")
    return result
