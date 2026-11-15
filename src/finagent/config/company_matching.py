"""Deterministic identification for the 51-company FinAgent-X universe.

Longest overlapping names win; separate companies remain separate candidates.
The scraper additionally requires a credible company name at the start of a block.
"""

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata

from .universe import universe


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    # D.I.Y. and DIY denote the same alias, not three separate words.
    value = re.sub(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?", lambda m: m[0].replace(".", ""), value)
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


# These are exclusions, not another investment universe. In particular, a
# subsidiary/REIT must not inherit its listed parent's short alias.
EXCLUDED_NAMES = (
    "Sunway Real Estate Investment Trust", "Sunway REIT",
    "Sunway Healthcare", "Sunway Medical Centre",
    "Genting Singapore", "Genting Hong Kong", "YTL Hospitality REIT",
)
_GENERIC_WORDS = {"time", "dialog", "bursa"}
_LEGAL_SUFFIX = re.compile(r"^(?:(?:group|holdings|corporation|corp|international)\s+)?(?:bhd|berhad)\b")
_SUBJECT_CONTINUATION = re.compile(
    r"^(?:s\b|has\b|have\b|had\b|is\b|are\b|was\b|were\b|will\b|"
    r"said\b|reported\b|posted\b|announced\b|recorded\b|secured\b|"
    r"rose\b|fell\b|plans\b|proposes\b|expects\b|shares\b|net\b|"
    r"revenue\b|profit\b|earnings\b|declared\b|received\b|acquired\b)"
)


@dataclass(frozen=True)
class CompanyResolution:
    company: str | None
    candidates: tuple[str, ...]
    status: str


@lru_cache(maxsize=1)
def _terms():
    terms = {}
    for company, info in universe.items():
        for name in (company, *info["aliases"]):
            terms.setdefault(normalize_name(name), set()).add(company)
    return terms


def _name_matches(text: str):
    normalized = normalize_name(text)
    matches = []
    for term, companies in _terms().items():
        for match in re.finditer(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
            if term in _GENERIC_WORDS and normalized != term:
                tail = normalized[match.end():].strip()
                if not _LEGAL_SUFFIX.match(tail):
                    continue
            matches.append((match.start(), match.end(), companies, term))
    for name in EXCLUDED_NAMES:
        term = normalize_name(name)
        for match in re.finditer(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
            matches.append((match.start(), match.end(), set(), term))
    # Suppress only overlapping shorter aliases. "Sunway and Sunway
    # Construction" still identifies two independent entities.
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[3]))
    kept = []
    for match in matches:
        if not any(match[0] >= other[0] and match[1] <= other[1] for other in kept):
            kept.append(match)
    return normalized, sorted(kept, key=lambda item: item[0])


def find_company_mentions(text: str) -> tuple[str, ...]:
    """Return all distinct candidates, never pick the first dictionary entry."""
    _, matches = _name_matches(text)
    companies = set().union(*(item[2] for item in matches)) if matches else set()
    for company, info in universe.items():
        if re.search(rf"(?<![\w.]){re.escape(info['ticker'])}(?![\w.])", text, re.I):
            companies.add(company)
    return tuple(sorted(companies))


def find_title_candidates(title: str) -> tuple[str, ...]:
    """Company-list headlines can contain exact generic aliases such as Dialog.

    Only exact comma/and/& separated names bypass the prose generic-word guard.
    Paragraph subjects must still pass the conservative entity matcher.
    """
    companies = set(find_company_mentions(title))
    for part in re.split(r"[,;|]|\s+(?:and|&)\s+", title, flags=re.I):
        companies.update(_terms().get(normalize_name(part), set()))
    return tuple(sorted(companies))


def resolve_company(text: str) -> CompanyResolution:
    normalized = normalize_name(text)
    exact = {company for company, info in universe.items()
             if str(text).strip().casefold() == info["ticker"].casefold()}
    if not exact:
        exact = _terms().get(normalized, set())
    candidates = tuple(sorted(exact)) if exact else find_company_mentions(text)
    if len(candidates) == 1:
        return CompanyResolution(candidates[0], candidates, "matched")
    if not candidates and any(not match[2] for match in _name_matches(text)[1]):
        return CompanyResolution(None, (), "outside_universe")
    return CompanyResolution(None, candidates, "ambiguous" if candidates else "not_found")


def identify_company(question: str) -> str | None:
    """Compatibility API: None means unknown OR ambiguous; use resolve_company for details."""
    return resolve_company(question).company


def identification_message(resolution: CompanyResolution) -> str:
    if resolution.status == "ambiguous":
        return "Please specify one company; multiple companies were identified: " + ", ".join(resolution.candidates) + "."
    if resolution.status == "outside_universe":
        return "The identified entity is outside the 51-company FinAgent-X universe. Please specify a covered company."
    return "I could not identify a company from the 51-company FinAgent-X universe."


def match_company_name(text: str) -> str | None:
    """Conservative paragraph subject matcher for scraping, not arbitrary mentions.

    An exact known name/alias is accepted. In prose, the name must be followed
    by a legal suffix, a possessive or a financial reporting predicate. Unknown
    name extensions (e.g. Sunway Medical...) are rejected, not assigned upward.
    """
    normalized, matches = _name_matches(text)
    leading = [m for m in matches if m[0] == 0]
    if len(leading) != 1 or len(leading[0][2]) != 1:
        return None
    _, end, companies, term = leading[0]
    tail = normalized[end:].strip()
    # Longest-match suppression can select an alias already ending in Bhd.
    # Its legal suffix is just as conclusive as a suffix following a short alias.
    complete_legal_name = re.search(r"\b(?:bhd|berhad)$", term)
    if tail and not (complete_legal_name or _LEGAL_SUFFIX.match(tail) or _SUBJECT_CONTINUATION.match(tail)):
        return None
    return next(iter(companies))
