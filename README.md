# FinAgent-X

FinAgent-X is a multi-agent financial intelligence platform for **51 Bursa Malaysia companies**.

It combines:

- **RAG Agent** for financial news analysis
- **Market Agent** for technical indicators
- **Forecast Agent** using an LSTM for next-session return prediction
- **LangGraph** for agent routing and synthesis
- **FastAPI** backend
- **Streamlit** dashboard with conversational memory

The dashboard includes market overview, company analysis, FinAgent AI chat, and model/system insights.

## Installation

Clone the repository:

```bash
git clone https://github.com/bilal-aouiss/FinAgent-X-An-Explainable-Multi-Agent-Financial-Intelligence-Platform.git FinAgent-X
cd FinAgent-X
```

Install dependencies with `uv`:

```bash
uv sync
```

Create:

```text
src/.env
```

and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_api_key
```

## Run the application

Open a first terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn finagent.api.main:app --reload --app-dir src
```

FastAPI will run at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

## Disclaimer

FinAgent-X is a research and portfolio project. Forecasts and technical indicators are experimental and should not be considered financial advice.
