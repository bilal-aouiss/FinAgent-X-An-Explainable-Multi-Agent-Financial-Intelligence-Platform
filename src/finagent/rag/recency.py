from dataclasses import dataclass
from datetime import date, timedelta
import calendar
import re


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date
    corpus_latest: date
    reason: str

    def describe(self) -> str:
        return (f"Publication window: {self.start} to {self.end}; "
                f"latest corpus date: {self.corpus_latest}.")


def parse_date(value) -> date:
    return date.fromisoformat(str(value)[:10])


def resolve_time_window(query: str, corpus_latest: date) -> DateWindow | None:
    query = " ".join(query.casefold().split())
    explicit = re.search(r"\b(?:from|between) (\d{4}-\d{2}-\d{2}) (?:to|and) (\d{4}-\d{2}-\d{2})\b", query)
    if explicit:
        start, end = map(date.fromisoformat, explicit.groups())
        if start > end:
            raise ValueError("The requested start date is after the end date.")
        return DateWindow(start, end, corpus_latest, "explicit_range")
    relative = re.search(r"\b(?:last|past) (\d+) (days?|weeks?)\b", query)
    if relative:
        days = int(relative[1]) * (7 if relative[2].startswith("week") else 1)
        if days < 1:
            raise ValueError("The requested lookback must be positive.")
        return DateWindow(corpus_latest - timedelta(days=days), corpus_latest, corpus_latest, "explicit_lookback")
    months = "january february march april may june july august september october november december".split()
    month = re.search(r"\b(?:in|during) (" + "|".join(months) + r") (\d{4})\b", query)
    if month:
        year, number = int(month[2]), months.index(month[1]) + 1
        return DateWindow(date(year, number, 1), date(year, number, calendar.monthrange(year, number)[1]), corpus_latest, "explicit_month")
    if re.search(r"\b(?:in|during) (?:" + "|".join(months) + r")\b", query):
        raise ValueError("Please specify the year for the requested month (for example September 2026).")
    quarter = re.search(r"\bq([1-4])(?:\s+(\d{4}))?\b", query)
    if quarter and (quarter[2] or re.search(r"\b(?:in|during)\s+q[1-4]\b", query)):
        if not quarter[2]:
            raise ValueError("Please specify the year for the requested quarter (for example Q2 2026).")
        year, first = int(quarter[2]), 3 * (int(quarter[1]) - 1) + 1
        last = first + 2
        return DateWindow(date(year, first, 1), date(year, last, calendar.monthrange(year, last)[1]), corpus_latest, "explicit_quarter")
    if re.search(r"\b(?:recent|recently|latest)\b|\blast weeks\b", query):
        return DateWindow(corpus_latest - timedelta(days=90), corpus_latest, corpus_latest, "recent_90_days")
    return None


def provenance_records(metadata: dict) -> list[dict]:
    return metadata.get("provenance") or [metadata]


def eligible_provenance(metadata: dict, window: DateWindow | None) -> list[dict]:
    records = provenance_records(metadata)
    if window is None:
        return records
    return [record for record in records if window.start <= parse_date(record["date"]) <= window.end]
