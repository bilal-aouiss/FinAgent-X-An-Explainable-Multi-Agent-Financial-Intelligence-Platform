import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import APIError, company_path, get
from ui import fetch, number, source_link

st.title("Company Intelligence")
st.caption("Market observations, technical context, financial news and the saved LSTM signal.")
companies = fetch("/companies")
names = [row["company"] for row in companies]
company = st.selectbox("Select Company", names, index=names.index("Maybank"), key="company_selection")
snapshot = fetch(company_path("market", company))
st.subheader(f"{company} · {snapshot['ticker']}")
st.caption(f"Market Snapshot · Observation date: {snapshot['date']}")
cols = [*st.columns(3), *st.columns(3)]
for col, label, field, digits, suffix in [
    (cols[0], "Price (MYR)", "close", 3, ""), (cols[1], "Daily Return", "daily_return", 2, "%"),
    (cols[2], "Volume", "volume", 0, ""), (cols[3], "RSI · 14", "rsi_14", 1, ""),
    (cols[4], "Momentum · 10 sessions", "momentum_10", 2, "%"),
    (cols[5], "Annualized Volatility · 20 sessions", "volatility_20", 2, "%"),
]:
    col.metric(label, number(snapshot.get(field), digits, suffix), border=True)
cols[2].caption(f"Volume change: {number(snapshot.get('volume_change'), 2, '%')}")

st.subheader("Price & Trend")
period = st.radio("Period", ["1M", "3M", "6M", "1Y", "MAX"], index=2, horizontal=True)
history = pd.DataFrame(fetch(company_path("company", company) + f"/history?period={period}"))


def has_history(*columns):
    return not history.empty and "date" in history and all(
        column in history and history[column].notna().any() for column in columns
    )


def chart_layout(fig, yaxis_title, height=300):
    fig.update_layout(
        height=height, yaxis_title=yaxis_title, hovermode="x unified",
        legend=dict(orientation="h", y=1.12), margin=dict(l=0, r=10, t=25, b=10),
    )
    return fig


if history.empty:
    st.info("No price history available for this selection.")
elif has_history("close"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.date, y=history.close, name="Close", line=dict(color="#45D5BF", width=2)))
    for column, label, color in [("sma_20", "SMA 20", "#7C9FFA"), ("sma_50", "SMA 50", "#DBB871"), ("sma_200", "SMA 200", "#C296ED")]:
        if has_history(column):
            fig.add_trace(go.Scatter(x=history.date, y=history[column], name=label, line=dict(color=color, width=1)))
    st.plotly_chart(chart_layout(fig, "Price (MYR)", height=390), width="stretch")
    missing = [label for field, label in [("sma_20", "SMA20"), ("sma_50", "SMA50"), ("sma_200", "SMA200")] if not has_history(field)]
    if missing:
        st.caption("Unavailable in this period: " + ", ".join(missing))
else:
    st.info("No closing prices are available for this selection.")
st.caption("All historical charts use the selected period, measured from the latest saved observation. Gaps indicate missing observations.")

st.subheader("Technical Intelligence")
st.caption("Trend, momentum and risk provide complementary descriptions. An isolated indicator is not an investment recommendation.")
macd_panel, rsi_panel = st.columns(2)
with macd_panel:
    st.markdown("**Trend · MACD**")
    if has_history("macd", "macd_signal", "macd_histogram"):
        fig = go.Figure()
        colors = ["#536079" if pd.isna(value) else "#45D5BF" if value >= 0 else "#E57B91" for value in history.macd_histogram]
        fig.add_trace(go.Bar(x=history.date, y=history.macd_histogram, name="Histogram", marker_color=colors, opacity=0.6))
        fig.add_trace(go.Scatter(x=history.date, y=history.macd, name="MACD", line=dict(color="#7C9FFA", width=2)))
        fig.add_trace(go.Scatter(x=history.date, y=history.macd_signal, name="Signal", line=dict(color="#DBB871", width=1.5)))
        fig.add_hline(y=0, line_width=1, line_color="#536079")
        st.plotly_chart(chart_layout(fig, "MACD (MYR)"), width="stretch")
        st.caption("MACD and its signal line describe trend; the histogram shows their difference.")
    else:
        st.info("MACD, signal or histogram history is unavailable for this period.")
with rsi_panel:
    st.markdown("**Momentum · RSI**")
    if has_history("rsi_14"):
        fig = go.Figure(go.Scatter(x=history.date, y=history.rsi_14, name="RSI 14", line=dict(color="#C296ED", width=2)))
        for threshold in [30, 70]:
            fig.add_hline(y=threshold, line_dash="dot", line_color="#8B97AD", annotation_text=str(threshold))
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(chart_layout(fig, "RSI · 14"), width="stretch")
        st.caption("30 and 70 are reference levels for momentum conditions, not predictions of a reversal.")
    else:
        st.info("RSI history is unavailable for this period.")

st.markdown("**Risk indicators**")
risk_cols = st.columns(3)
for col, label, field, digits, suffix in [
    (risk_cols[0], "ATR · 14 (MYR)", "atr_14", 4, ""),
    (risk_cols[1], "Annualized Volatility · 20 sessions", "volatility_20", 2, "%"),
    (risk_cols[2], "Bollinger Width", "bollinger_width", 2, "%"),
]:
    col.metric(label, number(snapshot.get(field), digits, suffix), border=True)
st.caption("ATR describes the absolute trading range; volatility describes return variability; Bollinger width describes relative price dispersion.")

with st.expander("Price dispersion · Bollinger Bands"):
    if has_history("close", "bollinger_upper", "bollinger_middle", "bollinger_lower"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.date, y=history.bollinger_lower, name="Lower", line=dict(color="#7C9FFA", width=1)))
        fig.add_trace(go.Scatter(x=history.date, y=history.bollinger_upper, name="Upper", line=dict(color="#7C9FFA", width=1),
                                 fill="tonexty", fillcolor="rgba(124,159,250,.12)"))
        fig.add_trace(go.Scatter(x=history.date, y=history.bollinger_middle, name="Middle", line=dict(color="#DBB871", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=history.date, y=history.close, name="Close", line=dict(color="#45D5BF", width=2)))
        st.plotly_chart(chart_layout(fig, "Price (MYR)"), width="stretch")
        st.caption("Bands describe the dispersion of prices around their moving average.")
    else:
        st.info("Bollinger band history is unavailable for this period.")

technical, forecast_panel = st.columns([1, 1])
with technical:
    st.subheader("Indicator details")
    rows = [
        ("Daily return", "daily_return", 2, "%"),
        ("SMA · 20", "sma_20", 3, " MYR"), ("SMA · 50", "sma_50", 3, " MYR"),
        ("SMA · 200", "sma_200", 3, " MYR"),
        ("EMA · 12", "ema_12", 3, " MYR"), ("EMA · 26", "ema_26", 3, " MYR"),
        ("MACD", "macd", 4, ""), ("MACD signal", "macd_signal", 4, ""),
        ("MACD histogram", "macd_histogram", 4, ""),
        ("RSI · 14", "rsi_14", 1, ""), ("10-session momentum", "momentum_10", 2, "%"),
        ("Annualized volatility · 20 sessions", "volatility_20", 2, "%"),
        ("ATR · 14", "atr_14", 4, " MYR"), ("Bollinger width", "bollinger_width", 2, "%"),
        ("Volume", "volume", 0, ""), ("Volume change", "volume_change", 2, "%"),
    ]
    with st.expander("Latest values · all dimensions"):
        st.dataframe(pd.DataFrame([{"Indicator": label, "Value": number(snapshot.get(field), digits, suffix)}
                                   for label, field, digits, suffix in rows]), hide_index=True, width="stretch")
    st.caption("Use FinAgent AI to request an interpretation of these observations.")

with forecast_panel:
    st.subheader("LSTM Forecast")
    with st.container(border=True):
        try:
            with st.spinner("Loading the saved LSTM signal…"):
                prediction = get(company_path("forecast", company))
            st.metric("Predicted next-session return", number(prediction["predicted_return"], 3, "%"))
            st.markdown(f"**Direction: {prediction['direction']}**")
            st.write(f"Last observation used: **{prediction['last_observation_date']}**")
            st.caption(f"Inference date (UTC): {prediction['prediction_date']}. Horizon: next trading session after the last observation.")
            st.info(prediction["message"])
        except APIError as exc:
            st.error(str(exc))

st.subheader("News Intelligence")
st.caption("Latest available company news. Sentiment applies to the company-specific passage.")
try:
    articles = get(company_path("company", company) + "/news")
    if not articles:
        st.info("No news is available for this company in the saved corpus.")
    for item in articles:
        with st.container(border=True):
            st.markdown(f"**{item.get('title') or 'Financial news'}**")
            st.caption(f"{item.get('date') or 'Date unavailable'} · {(item.get('sentiment') or 'unclassified').title()} · {item.get('source') or ''}")
            with st.expander("Company passage"):
                st.write(item.get("text") or "No excerpt available.")
            source_link(item.get("url"))
except APIError as exc:
    st.warning(str(exc))
