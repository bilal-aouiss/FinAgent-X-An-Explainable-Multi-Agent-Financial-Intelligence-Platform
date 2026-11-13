import os

from getpass import getpass

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .data import (
    identify_company,
    get_latest_market_data
)


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()


if not os.getenv(
    "GOOGLE_API_KEY"
):

    os.environ[
        "GOOGLE_API_KEY"
    ] = getpass(
        "Google API Key: "
    )


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# 3. MARKET PROMPT
# ============================================================

market_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Market Analysis Agent of FinAgent-X.

You analyze market data and technical indicators
for Malaysian listed companies.

Use ONLY the market data provided to you.

Rules:

1. Do not invent prices or indicators.

2. Always mention the date of the market observation.

3. Answer the user's specific question.

4. Be concise and financially precise.

5. Technical indicators are signals, not guarantees
   of future price movements.

6. Do not make a forecast unless a forecast model
   explicitly provides one.

7. If the available market data is insufficient,
   say so clearly.
"""
        ),

        (
            "human",
            """
QUESTION:

{question}


COMPANY:

{company}


LATEST MARKET DATA:

Date: {date}

Close: {close}

Daily return (%): {daily_return}

Intraday return (%): {intraday_return}

Daily range (%): {daily_range}

Volume: {volume}

Volume change (%): {volume_change}

RSI 14: {rsi_14}

SMA 20: {sma_20}

SMA 50: {sma_50}

SMA 200: {sma_200}

EMA 12: {ema_12}

EMA 26: {ema_26}

MACD: {macd}

MACD signal: {macd_signal}

MACD histogram: {macd_histogram}

ATR 14: {atr_14}

Bollinger upper: {bollinger_upper}

Bollinger middle: {bollinger_middle}

Bollinger lower: {bollinger_lower}

Bollinger width (%): {bollinger_width}

Annualized volatility (%): {volatility_20}

10-day momentum (%): {momentum_10}


ANSWER:
"""
        )
    ]
)


# ============================================================
# 4. GENERATION CHAIN
# ============================================================

market_chain = (
    market_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 5. MARKET ANALYSIS PIPELINE
# ============================================================

def analyze_market(
    question
):

    # --------------------------------------------------------
    # 1. Identify company
    # --------------------------------------------------------

    company = identify_company(
        question
    )

    if company is None:

        return {
            "company": None,
            "answer": (
                "I could not identify a company "
                "from the FinAgent-X universe."
            ),
            "market_data": None
        }


    # --------------------------------------------------------
    # 2. Get latest observation
    # --------------------------------------------------------

    latest = get_latest_market_data(
        company
    )

    if latest is None:

        return {
            "company": company,
            "answer": (
                f"No market data is available "
                f"for {company}."
            ),
            "market_data": None
        }


    # --------------------------------------------------------
    # 3. Ask the LLM to interpret the data
    # --------------------------------------------------------

    answer = market_chain.invoke(
        {
            "question":
                question,

            "company":
                company,

            "date":
                latest["date"],

            "close":
                latest["close"],

            "daily_return":
                latest["daily_return"],

            "intraday_return":
                latest["intraday_return"],

            "daily_range":
                latest["daily_range"],

            "volume":
                latest["volume"],

            "volume_change":
                latest["volume_change"],

            "rsi_14":
                latest["rsi_14"],

            "sma_20":
                latest["sma_20"],

            "sma_50":
                latest["sma_50"],

            "sma_200":
                latest["sma_200"],

            "ema_12":
                latest["ema_12"],

            "ema_26":
                latest["ema_26"],

            "macd":
                latest["macd"],

            "macd_signal":
                latest["macd_signal"],

            "macd_histogram":
                latest["macd_histogram"],

            "atr_14":
                latest["atr_14"],

            "bollinger_upper":
                latest["bollinger_upper"],

            "bollinger_middle":
                latest["bollinger_middle"],

            "bollinger_lower":
                latest["bollinger_lower"],

            "bollinger_width":
                latest["bollinger_width"],

            "volatility_20":
                latest["volatility_20"],

            "momentum_10":
                latest["momentum_10"]
        }
    )


    return {
        "company":
            company,

        "answer":
            answer,

        "market_data":
            latest.to_dict()
    }