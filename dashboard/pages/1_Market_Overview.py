import pandas as pd
import plotly.express as px
import streamlit as st

from ui import fetch, number

st.title("FinAgent-X")
st.markdown("Malaysian Multi-Agent Financial Intelligence Platform")
data = pd.DataFrame(fetch("/market"))
if data.empty:
    st.info("No market observations are available.")
    st.stop()
for field in ["daily_return", "momentum_10", "rsi_14", "volatility_20"]:
    data[field] = pd.to_numeric(data[field], errors="coerce")

st.caption(f"MARKET OVERVIEW · {len(data)} Companies · Latest available observations: {data['date'].min()} — {data['date'].max()}")
st.caption("Performance · Trend · Momentum · Risk · Price dispersion · Volume")
cards = st.columns(4)
for col, label, field, highest in [
    (cards[0], "Top Gainer", "daily_return", True),
    (cards[1], "Top Loser", "daily_return", False),
    (cards[2], "Strongest Momentum", "momentum_10", True),
    (cards[3], "Highest Volatility", "volatility_20", True),
]:
    available = data.dropna(subset=[field])
    if field == "daily_return":
        available = available[available[field] > 0] if highest else available[available[field] < 0]
    with col:
        if available.empty:
            st.metric(label, "None" if field == "daily_return" else "Unavailable", border=True)
        else:
            row = available.loc[available[field].idxmax() if highest else available[field].idxmin()]
            st.metric(label, number(row[field], suffix="%"), border=True)
            st.caption(row["company"])

# Compact secondary row keeps RSI in proportion to the other market dimensions.
for col, label, overbought in zip(st.columns(2), ["Most Overbought", "Most Oversold"], [True, False]):
    available = data[data.rsi_14 > 70] if overbought else data[data.rsi_14 < 30]
    with col, st.container(border=True):
        if available.empty:
            st.write(f"**{label}** · No company {'above 70' if overbought else 'below 30'}")
        else:
            row = available.loc[available.rsi_14.idxmax() if overbought else available.rsi_14.idxmin()]
            st.write(f"**{label}** · {row.company} · RSI {number(row.rsi_14, 1)}")

# Thresholds always use the full current universe, before search or other filters.
return_strength = data.daily_return.abs().quantile(0.75)
momentum_strength = data.momentum_10.abs().quantile(0.75)
volatility_low, volatility_high = data.volatility_20.quantile([1 / 3, 2 / 3])

st.subheader("Company universe")
search = st.text_input("Search company or ticker", placeholder="Maybank, CIMB, 1155.KL…")
performance_col, momentum_col, rsi_col, volatility_col = st.columns(4)
performance = performance_col.selectbox("Performance", ["All", "Gainers", "Losers", "Strong gainers", "Strong losers"],
    help="Strong moves have an absolute daily return at or above the universe's 75th percentile.")
momentum = momentum_col.selectbox("Momentum", ["All", "Positive", "Negative", "Strong positive", "Strong negative"],
    help="10-session momentum. Strong moves have an absolute value at or above the universe's 75th percentile.")
rsi = rsi_col.selectbox("RSI", ["All", "Oversold (<30)", "Neutral (30–70)", "Overbought (>70)"])
volatility = volatility_col.selectbox("Volatility", ["All", "Low", "Medium", "High"],
    help="Annualized volatility grouped using the current universe's 33⅓ and 66⅔ percentiles.")
with st.expander("Filter definitions and current thresholds"):
    strength_text = lambda value: f"{value:.4f}%" if pd.notna(value) else "unavailable"
    st.write(
        "Strong performance/momentum: the move must have the selected sign and an absolute "
        "value at or above the 75th percentile of absolute values across the full universe. "
        f"Current cutoffs: daily return ≈ {strength_text(return_strength)}; "
        f"10-session momentum ≈ {strength_text(momentum_strength)}."
    )
    st.write(
        f"Volatility tercile boundaries: ≈ {strength_text(volatility_low)} and {strength_text(volatility_high)}. "
        "Low ≤ first boundary; Medium > first and ≤ second; High > second. "
        "Equal values stay together, so groups may differ in size."
    )
    st.write("RSI: Oversold <30; Neutral includes 30 and 70; Overbought >70. Missing values are excluded only when their filter is active.")
    st.caption("Thresholds use all available observations in the 51-company snapshot and do not change when other filters are applied. These are descriptive groups, not investment recommendations.")
view = data.copy()
if search:
    view = view[view["company"].str.contains(search, case=False, regex=False) | view["ticker"].str.contains(search, case=False, regex=False)]
if performance in {"Gainers", "Strong gainers"}:
    view = view[view.daily_return > 0]
elif performance in {"Losers", "Strong losers"}:
    view = view[view.daily_return < 0]
if performance.startswith("Strong"):
    view = view[view.daily_return.abs() >= return_strength]
if momentum in {"Positive", "Strong positive"}:
    view = view[view.momentum_10 > 0]
elif momentum in {"Negative", "Strong negative"}:
    view = view[view.momentum_10 < 0]
if momentum.startswith("Strong"):
    view = view[view.momentum_10.abs() >= momentum_strength]
if rsi.startswith("Overbought"):
    view = view[view.rsi_14 > 70]
elif rsi.startswith("Oversold"):
    view = view[view.rsi_14 < 30]
elif rsi.startswith("Neutral"):
    view = view[view.rsi_14.between(30, 70)]
if volatility == "Low":
    view = view[view.volatility_20 <= volatility_low]
elif volatility == "Medium":
    view = view[(view.volatility_20 > volatility_low) & (view.volatility_20 <= volatility_high)]
elif volatility == "High":
    view = view[view.volatility_20 > volatility_high]

columns = ["company", "ticker", "date", "close", "daily_return", "rsi_14", "momentum_10", "volatility_20", "macd"]
st.dataframe(view.sort_values("daily_return", ascending=False)[columns], hide_index=True,
    width="stretch", height=520, column_config={
        "company": "Company", "ticker": "Ticker", "date": "Observation date",
        "close": st.column_config.NumberColumn("Price (MYR)", format="%.3f"),
        "daily_return": st.column_config.NumberColumn("Daily return (%)", format="%.2f"),
        "rsi_14": st.column_config.NumberColumn("RSI · 14", format="%.1f"),
        "momentum_10": st.column_config.NumberColumn("Momentum · 10 (%)", format="%.2f"),
        "volatility_20": st.column_config.NumberColumn("Volatility · annualized (%)", format="%.2f"),
        "macd": st.column_config.NumberColumn("MACD", format="%.4f"),
    })
st.caption(f"Showing {len(view)} of {len(data)} companies. Click any column header to sort. Missing values are left empty.")
if view.empty:
    st.info("No companies match these filters. Broaden a filter or clear the search.")

st.subheader("Market dimensions")
st.caption("Charts reflect the selected companies. Snapshot cards above summarize the full universe.")
left, right = st.columns([3, 2])
with left:
    st.markdown("**Daily performance · Top gainers / losers**")
    movers = pd.concat([
        view[view.daily_return < 0].nsmallest(5, "daily_return"),
        view[view.daily_return > 0].nlargest(5, "daily_return"),
    ]).sort_values("daily_return")
    if movers.empty:
        st.info("No gainers or losers with available returns in this selection.")
    else:
        fig = px.bar(movers, x="daily_return", y="company", orientation="h", color="daily_return",
                     color_continuous_scale=["#E57B91", "#26354B", "#45D5BF"], color_continuous_midpoint=0,
                     labels={"daily_return": "Daily return (%)", "company": ""})
        fig.update_layout(height=360, showlegend=False, coloraxis_showscale=False, margin=dict(l=0, r=10, t=5, b=0))
        st.plotly_chart(fig, width="stretch")
with right:
    st.markdown("**Momentum vs Volatility**")
    points = view.dropna(subset=["momentum_10", "volatility_20"])
    if points.empty:
        st.info("No momentum and volatility observations in this selection.")
    else:
        fig = px.scatter(points, x="momentum_10", y="volatility_20", hover_name="company",
                         hover_data={"ticker": True, "rsi_14": ":.1f", "daily_return": ":.2f", "momentum_10": ":.2f", "volatility_20": ":.2f"},
                         labels={"momentum_10": "Momentum · 10 (%)", "volatility_20": "Annualized volatility (%)", "rsi_14": "RSI 14", "daily_return": "Daily return (%)"})
        fig.update_traces(marker=dict(size=9, color="#7C9FFA", opacity=0.8))
        fig.add_vline(x=0, line_dash="dot", line_color="#536079")
        fig.update_layout(height=360, margin=dict(l=0, r=10, t=5, b=0))
        st.plotly_chart(fig, width="stretch")
st.caption("Momentum uses 10 sessions; annualized volatility uses a 20-session window. Indicators describe market conditions and do not predict outcomes on their own.")
