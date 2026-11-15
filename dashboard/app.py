from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import APIError, get

st.set_page_config(page_title="FinAgent-X", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -.025em;}
[data-testid="stMetricValue"] {font-size: 1.7rem;}
[data-testid="stSidebar"] {border-right: 1px solid #26354b;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ◈ FinAgent-X")
    st.caption("MALAYSIAN FINANCIAL INTELLIGENCE")
    st.divider()

page = st.navigation([
    st.Page("pages/1_Market_Overview.py", title="Market Overview", icon=":material/monitoring:", default=True),
    st.Page("pages/2_Company_Intelligence.py", title="Company Intelligence", icon=":material/business:", url_path="company"),
    st.Page("pages/3_FinAgent_AI.py", title="FinAgent AI", icon=":material/forum:", url_path="chat"),
    st.Page("pages/4_Model_System_Insights.py", title="Model & System Insights", icon=":material/account_tree:", url_path="insights"),
])

with st.sidebar:
    st.divider()
    try:
        get("/health")
        st.caption("🟢 Backend connected")
    except APIError:
        st.caption("🔴 Backend unavailable")
    st.caption("51 companies · Bursa Malaysia")
    st.caption("Saved market observations and experimental AI analysis.")
    if st.button("Refresh data", width="stretch"):
        get.clear()
        st.rerun()
    st.caption("Experimental quantitative signals. Not investment advice.")

page.run()
