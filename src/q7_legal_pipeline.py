"""Question 7 — Local-first legal transcription pipeline (Hindi / English / Hinglish).

Deterministic, offline, standard-library only:

    simulated STT -> transcript correction -> relevance filtering
    -> legal extraction against a versioned local DB -> benchmark metrics

Guarantees:
  * No model downloads, no network calls, no GPU, no API spend.
  * Negations ("not" / "nahi" / Devanagari "नहीं") are never dropped.
  * Legal sections are never invented: an unknown section is flagged as
    "not in local DB v1.0 — verify manually".
"""

from __future__ import annotations

import re
import time
from typing import Any

# ---------------------------------------------------------------------------
# Relevance labels
# ---------------------------------------------------------------------------
LEGALLY_RELEVANT = "LEGALLY_RELEVANT"
IRRELEVANT = "IRRELEVANT"
UNCERTAIN = "UNCERTAIN"

# ---------------------------------------------------------------------------
# Versioned local legal database — the ONLY source of legal provisions.
# Never invent entries at runtime.
# ---------------------------------------------------------------------------
DB_VERSION = "1.0"
LEGAL_DB: dict[str, dict[str, str]] = {
    "Section 73": {"act": "Indian Contract Act", "note": "compensation for breach"},
    "Section 420": {"act": "IPC (old)", "note": "cheating — verify against BNS mapping"},
    "Section 138": {"act": "Negotiable Instruments Act", "note": "cheque dishonour"},
}

# ---------------------------------------------------------------------------
# Simulated STT fixtures (fixed, in-code, no acoustic model).
# ---------------------------------------------------------------------------
AUDIO_FIXTURES: dict[str, dict[str, Any]] = {
    "en_agreement_01": {
        "language": "en",
        "wer_hint": 0.12,
        "transcript": (
            "Good morning everyone, the weather is pleasant today.\n"
            "Sharma versus Verma signed an agreement on 12 January 2023.\n"
            "The defendant did not pay the amount twlce as agreed.\n"
            "This is a clear breach of the contract.\n"
            "As per section seventy three the claimant can seek compensation.\n"
            "The client also mentioned Section 999 which we must verify.\n"
            "After the meeting we watched a movie."
        ),
    },
    "hi_payment_01": {
        "language": "hinglish",
        "wer_hint": 0.18,
        "transcript": (
            "नमस्ते सर, आज मौसम अच्छा है।\n"
            "कल की बात है कि भुगतान 15 March 2024 तक नहीं हुआ।\n"
            "समझौता के अनुसार धारा 73 लागू हो सकती है।\n"
            "बाद में movie देखने चलेंगे।"
        ),
    },
    "en_chitchat_01": {
        "language": "en",
        "wer_hint": 0.05,
        "transcript": (
            "Hey how are you doing today.\n"
            "The weather is really nice this weekend.\n"
            "I tried a new food place yesterday.\n"
            "Did you watch the cricket match last night.\n"
            "The movie was amazing."
        ),
    },
}


class SimulatedSTT:
    """Deterministic offline stand-in for a speech-to-text model."""

    def __init__(self, fixtures: dict[str, dict[str, Any]] | None = None) -> None:
        self._fixtures = fixtures if fixtures is not None else AUDIO_FIXTURES

    def transcribe(self, audio_id: str) -> dict[str, Any]:
        if audio_id not in self._fixtures:
            raise KeyError(f"unknown audio_id: {audio_id}")
        fixture = self._fixtures[audio_id]
        return {
            "transcript": fixture["transcript"],
            "language": fixture["language"],
            "wer_hint": fixture["wer_hint"],
        }


# ---------------------------------------------------------------------------
# Transcript correction (deterministic; negations preserved by construction)
# ---------------------------------------------------------------------------
_ASR_CORRECTIONS: list[tuple[str, str]] = [
    ("section seventy three", "Section 73"),
    ("section four twenty", "Section 420"),
    ("section one thirty eight", "Section 138"),
    ("dhara seventy three", "Section 73"),
    ("twlce", "twice"),
    ("paymnt", "payment"),
    ("agreemnt", "agreement"),
    ("samjauta", "samjhauta"),
]


def correct_transcript(text: str) -> str:
    """Apply fixed ASR-error replacements and whitespace normalisation.

    Only substitutions are performed; no token is ever removed, so negations
    such as "not", "nahi" and "नहीं" survive untouched.
    """
    out = text
    for wrong, right in _ASR_CORRECTIONS:
        out = re.sub(re.escape(wrong), right, out, flags=re.IGNORECASE)
    out = re.sub(r"[ \t]+", " ", out)
    lines = [line.strip() for line in out.split("\n")]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Legal relevance classification (keyword rules, prefix-matched)
# ---------------------------------------------------------------------------
_RELEVANT_ASCII: list[str] = [
    "section", "agreement", "agree", "contract", "payment", "pay", "paid",
    "amount", "breach", "court", "police", "witness", "allegation", "evidence",
    "compensation", "claim", "clause", "provision", "notice", "damages",
    "defendant", "plaintiff", "claimant", "accused", "liability", "dispute",
    "cheque", "dishonour", "fraud", "cheat", "date", "dhara", "samjhauta",
    "bhugtan", "tareekh", "adalat", "dawa", "gawah",
]
_RELEVANT_NONASCII: list[str] = [
    "धारा", "समझौता", "भुगतान", "तारीख", "अदालत", "दावा", "गवाह",
]
_IRRELEVANT_ASCII: list[str] = [
    "weather", "food", "movie", "cricket", "weekend", "song", "lunch",
    "dinner", "match", "how are you", "mausam", "khana", "gaana",
]
_IRRELEVANT_NONASCII: list[str] = ["मौसम", "खाना", "फिल्म", "गाना", "क्रिकेट"]


def _match_ascii(text_lower: str, keywords: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(kw)}", text_lower) for kw in keywords)


def _match_substr(text_lower: str, keywords: list[str]) -> bool:
    return any(kw in text_lower for kw in keywords)


def classify_segment(segment: str) -> str:
    """Label one segment; legally relevant facts win over small talk."""
    text = segment.lower()
    if _match_ascii(text, _RELEVANT_ASCII) or _match_substr(text, _RELEVANT_NONASCII):
        return LEGALLY_RELEVANT
    if _match_ascii(text, _IRRELEVANT_ASCII) or _match_substr(text, _IRRELEVANT_NONASCII):
        return IRRELEVANT
    return UNCERTAIN


def filter_transcript(
    segments: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (kept, dropped): keep relevant + uncertain, drop irrelevant."""
    kept: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for segment in segments:
        label = classify_segment(segment)
        item = {"text": segment, "label": label}
        (dropped if label == IRRELEVANT else kept).append(item)
    return kept, dropped


# ---------------------------------------------------------------------------
# Legal extraction against the versioned local DB
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"\b(?:Section|Dhara|धारा)\s*(\d{1,4}[A-Za-z]?)", re.IGNORECASE
)
_DATE_RE = re.compile(r"\b\d{1,2}\s+[A-Za-z\u0900-\u097F]+\s+\d{4}\b")
_PARTY_RE = re.compile(r"\b([A-Z][a-z]+)\s+(?:vs|versus|and)\s+([A-Z][a-z]+)\b")

_LEGAL_TERMS: list[str] = [
    "agreement", "contract", "breach", "compensation", "damages", "allegation",
    "evidence", "witness", "negligence", "cheque", "dishonour", "fraud",
    "cheating", "court", "police", "clause", "provision", "notice", "claim",
    "defendant", "plaintiff", "accused", "samjhauta", "dhara", "adalat",
    "dawa", "gawah", "समझौता", "धारा", "अदालत", "दावा", "गवाह",
]


def _norm_section(number: str) -> str:
    return f"Section {number.upper()}"


def find_sections(text: str) -> list[str]:
    """Explicitly mentioned sections, normalised to e.g. "Section 73"."""
    found: list[str] = []
    for match in _SECTION_RE.finditer(text):
        key = _norm_section(match.group(1))
        if key not in found:
            found.append(key)
    return found


def lookup_section(section: str) -> dict[str, Any]:
    """DB lookup only. Unknown sections are flagged, never invented."""
    key = section if section.startswith("Section") else _norm_section(section)
    if key in LEGAL_DB:
        record = LEGAL_DB[key]
        return {
            "section": key,
            "known": True,
            "act": record["act"],
            "note": record["note"],
            "status": "STATED",
        }
    return {
        "section": key,
        "known": False,
        "warning": f"not in local DB v{DB_VERSION} — verify manually",
        "status": "STATED",
    }


def _suggest_keys(text: str) -> list[str]:
    """Keyword heuristics returning candidate DB keys (never free text)."""
    low = text.lower()

    def present(*words: str) -> bool:
        return any(word in low for word in words)

    keys: list[str] = []
    if present("payment", "pay", "bhugtan", "भुगतान") and present(
        "breach", "dishonour", "cheque", "bounced", "default"
    ):
        keys.append("Section 73")
    if present("cheque", "dishonour", "bounced") and present(
        "payment", "bhugtan", "भुगतान", "bank"
    ):
        keys.append("Section 138")
    if present("cheating", "fraud", "fraudster"):
        keys.append("Section 420")
    return keys


def extract_legal_info(transcript: str) -> dict[str, Any]:
    """Extract parties, dates, mentioned sections, DB provisions and terms."""
    sections = find_sections(transcript)
    stated = [lookup_section(section) for section in sections]
    stated_keys = {entry["section"] for entry in stated}

    suggested: list[dict[str, Any]] = []
    for key in _suggest_keys(transcript):
        if key in stated_keys:
            continue
        record = LEGAL_DB[key]
        suggested.append(
            {
                "section": key,
                "known": True,
                "act": record["act"],
                "note": record["note"],
                "status": "SUGGESTED",
            }
        )

    parties: list[str] = []
    for match in _PARTY_RE.finditer(transcript):
        party = re.sub(r"\s+", " ", match.group(0)).strip()
        if party not in parties:
            parties.append(party)

    terms: list[str] = []
    low = transcript.lower()
    for term in _LEGAL_TERMS:
        if term.isascii():
            if re.search(rf"\b{re.escape(term)}", low) and term not in terms:
                terms.append(term)
        elif term in low and term not in terms:
            terms.append(term)
    terms.sort()

    return {
        "parties": parties,
        "dates": _DATE_RE.findall(transcript),
        "sections_mentioned": [entry["section"] for entry in stated],
        "suggested_provisions": stated + suggested,
        "legal_terms": terms,
    }


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def split_segments(transcript: str) -> list[str]:
    return [line.strip() for line in transcript.split("\n") if line.strip()]


def run_pipeline(audio_id: str) -> dict[str, Any]:
    """Run STT -> correction -> filtering -> extraction and time the result."""
    start = time.perf_counter()
    raw = SimulatedSTT().transcribe(audio_id)
    corrected = correct_transcript(raw["transcript"])
    segments = split_segments(corrected)
    labeled = [{"text": s, "label": classify_segment(s)} for s in segments]
    kept, _dropped = filter_transcript(segments)
    relevant_only = [item["text"] for item in kept]
    extraction = extract_legal_info(corrected)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "audio_id": audio_id,
        "language": raw["language"],
        "corrected_transcript": corrected,
        "segments": labeled,
        "relevant_only": relevant_only,
        "extraction": extraction,
        "metrics": {
            "latency_ms": round(latency_ms, 4),
            "wer_proxy": raw["wer_hint"],
            "api_cost_usd": 0.0,
            "cpu_note": "local deterministic rules; no model download, no network, no GPU",
        },
    }


def benchmark() -> list[dict[str, Any]]:
    """Run every fixture, print a table, and return the rows."""
    rows: list[dict[str, Any]] = []
    for audio_id in AUDIO_FIXTURES:
        result = run_pipeline(audio_id)
        segments = result["segments"]
        kept = [seg for seg in segments if seg["label"] != IRRELEVANT]
        ratio = len(kept) / len(segments) if segments else 0.0
        metrics = result["metrics"]
        rows.append(
            {
                "audio_id": audio_id,
                "latency_ms": metrics["latency_ms"],
                "wer_proxy": metrics["wer_proxy"],
                "relevant_ratio": round(ratio, 3),
                "api_cost_usd": metrics["api_cost_usd"],
            }
        )

    print(f"{'audio_id':<18}{'latency_ms':>12}{'wer_proxy':>11}{'rel_ratio':>11}{'cost_usd':>10}")
    for row in rows:
        print(
            f"{row['audio_id']:<18}{row['latency_ms']:>12.4f}"
            f"{row['wer_proxy']:>11.2f}{row['relevant_ratio']:>11.3f}"
            f"{row['api_cost_usd']:>10.2f}"
        )
    return rows


if __name__ == "__main__":
    import json

    benchmark()
    print("\nFull pipeline (en_agreement_01) — offline, cost $0.00:")
    print(json.dumps(run_pipeline("en_agreement_01"), ensure_ascii=False, indent=2))
