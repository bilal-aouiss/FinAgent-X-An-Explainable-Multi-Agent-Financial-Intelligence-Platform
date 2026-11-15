import pandas as pd
import plotly.express as px
import streamlit as st

from ui import fetch

st.title("Model & System Insights")
st.caption("How FinAgent-X works · Existing experimental project results")
metrics = fetch("/insights")

st.subheader("Architecture")
st.graphviz_chart('''digraph {
 graph [rankdir=LR, bgcolor="transparent"];
 node [shape=box, style="rounded,filled", fillcolor="#151F32", color="#45D5BF", fontcolor="#E6EDF7", fontname="Arial"];
 edge [color="#7C9FFA"];
 User -> Streamlit -> FastAPI -> LangGraph -> Router;
 Memory [label="RAM conversation history"];
 FastAPI -> Memory [dir=both];
 Router -> {RAG Market Forecast};
 {RAG Market Forecast} -> Synthesis -> Answer;
 Answer [label="Final answer → API → UI"];
}''', width="stretch")
st.write("The router selects the agents needed for the current question. Each turn starts with a fresh LangGraph state; RAM history supplies conversational context. Synthesis combines the newly computed agent outputs.")

st.subheader("RAG Pipeline")
st.graphviz_chart('''digraph {
 graph [rankdir=LR, bgcolor="transparent"];
 node [shape=box, style="rounded,filled", fillcolor="#151F32", color="#7C9FFA", fontcolor="#E6EDF7", fontname="Arial"];
 edge [color="#45D5BF"];
 "Financial News" -> Documents -> "MPNet Embeddings" -> FAISS -> "CrossEncoder Reranking" -> "Top Documents" -> Gemini -> "Grounded Answer";
}''', width="stretch")

st.subheader("Retrieval Evaluation")
st.caption(f"Official project benchmark · {metrics['retrieval_questions']} questions · Existing experimental results")
retrieval = pd.DataFrame(metrics["retrieval"])
st.dataframe(retrieval, hide_index=True, width="stretch",
             column_config={key: st.column_config.NumberColumn(key, format="%.6f") for key in ["Hit@5", "Precision@5", "Recall@5", "MRR@5"]})
fig = px.bar(retrieval.melt(id_vars="Method", var_name="Metric", value_name="Score"),
             x="Metric", y="Score", color="Method", barmode="group", range_y=[0, 1],
             color_discrete_sequence=["#7C9FFA", "#DBB871", "#45D5BF"])
fig.update_layout(height=330, legend=dict(orientation="h", y=1.15), margin=dict(l=0, r=0, t=25, b=0))
st.plotly_chart(fig, width="stretch")
st.success("FAISS + CrossEncoder Reranker achieved the strongest retrieval performance.")

st.subheader("RAG Generation Evaluation")
for column, (label, value) in zip(st.columns(3), metrics["generation"].items()):
    column.metric(label, f"{value:.3f}", border=True)
st.caption("Existing LLM-as-a-Judge evaluation results.")

st.subheader("Forecasting Results")
forecast = pd.DataFrame(metrics["forecasting"])
st.dataframe(forecast, hide_index=True, width="stretch",
             column_config={key: st.column_config.NumberColumn(key, format="%.6f") for key in ["MAE", "RMSE", "Directional Accuracy"]})
st.caption("Saved outputs of the latest manual four-model comparison in forecasting_models.ipynb. MAE/RMSE are in percentage points; directional accuracy is a proportion.")
st.info("Final prototype model: LSTM, selected and saved after the latest manual training. The naive baseline remains competitive; the LSTM has the lowest RMSE in this comparison.")

with st.expander("Hyperparameter Optimization Experiment"):
    st.write("TTAO was investigated as a hyperparameter optimization strategy for the LSTM architecture. Its computational cost was high relative to the observed gains. The final prototype uses the LSTM selected during the latest manual training.")

st.subheader("Limitations")
st.markdown("""
- Financial returns are inherently noisy.
- The naive baseline remains competitive.
- Forecasts are experimental.
- RAG quality depends on corpus coverage and freshness.
- News sentiment may not translate directly into price movements.
""")
