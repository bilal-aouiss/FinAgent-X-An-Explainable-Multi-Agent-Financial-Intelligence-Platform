from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("message")
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank.")
        return value


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    selected_agents: list[Literal["rag", "market", "forecast"]]
    sources: list[dict] = Field(default_factory=list)


class Company(BaseModel):
    company: str
    ticker: str


class ForecastResponse(Company):
    prediction_date: str = Field(description="UTC date when inference was performed, not a future trading date.")
    last_observation_date: str
    horizon: str = "next trading session after last observation"
    predicted_return: float = Field(description="Percentage points, e.g. 0.5 means +0.5%.")
    direction: Literal["UP", "DOWN", "FLAT"]
    model: Literal["LSTM"] = "LSTM"
    message: str = "Experimental model-based quantitative signal. Not investment advice."
