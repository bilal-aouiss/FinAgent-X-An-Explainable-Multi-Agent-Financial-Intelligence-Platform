"""Process-local conversation history; graph outputs are never stored as state."""

from dataclasses import dataclass, field
from threading import Lock, RLock
import re

from finagent.config.company_matching import resolve_company

MAX_MESSAGES = 20
MAX_SESSIONS = 200
_reference = re.compile(r"\b(its|it|this company|that company|elle|celle-ci|son|ses)\b", re.I)


@dataclass
class Conversation:
    history: list[dict[str, str]] = field(default_factory=list)
    current_company: str | None = None
    lock: RLock = field(default_factory=RLock, repr=False)

    def fresh_state(self, message):
        resolution = resolve_company(message)
        companies = list(resolution.candidates)
        question = message
        if self.current_company and resolution.status != "outside_universe":
            comparison = re.search(r"\b(compare|comparison|versus|vs|comparer|comparez)\b", message, re.I)
            if _reference.search(message) and (not companies or comparison):
                question = _reference.sub(self.current_company, message)
                if self.current_company not in companies:
                    companies.insert(0, self.current_company)
            elif not companies and re.match(r"^(what about|how about|and |et |quel|quelle)", message, re.I):
                companies = [self.current_company]
                question = f"{message}\nCompany context: {self.current_company}"
        return {
            "question": question,
            "companies": companies,
            "conversation_history": [dict(m) for m in self.history],
        }

    def remember(self, message, answer, companies):
        # Store plain conversation, never rag_answer/market_answer/forecast_answer/state.
        self.history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer[:12000]},
        ])
        self.history[:] = self.history[-MAX_MESSAGES:]
        if len(companies) == 1:
            self.current_company = companies[0]
        elif companies and self.current_company not in companies:
            self.current_company = None


sessions: dict[str, Conversation] = {}
_sessions_lock = Lock()


def get_session(session_id):
    with _sessions_lock:
        if session_id not in sessions:
            if len(sessions) >= MAX_SESSIONS:
                raise RuntimeError("Conversation capacity reached. Restart the prototype to clear RAM sessions.")
            sessions[session_id] = Conversation()
        return sessions[session_id]
