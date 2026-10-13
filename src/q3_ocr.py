"""Question 3 — OCR review with confidence gating and safe corrections.

Validation strategy
-------------------
Every OCR field carries a confidence score and behaviour is gated on it:

* ``>= 0.90`` -> ``HIGH_CONFIDENCE``: a correction may be *applied*.
* ``0.60 .. 0.89`` -> ``REVIEW_REQUIRED``: a correction is only *suggested*
  together with a reason and is never applied automatically.
* ``< 0.60`` -> ``LOW_CONFIDENCE``: never auto-correct; flag for a human.

The original OCR text is always preserved (``CorrectionRecord.original``).
``clean_text`` only strips and collapses whitespace and never rewrites
characters; digit/letter fixes are emitted as suggestions only.

Medicine names are validated against a trusted dictionary: an exact
case-insensitive match first, then a ``difflib.get_close_matches`` fallback
with an explicit similarity cutoff.  Fuzzy hits are always marked
``needs_review``.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

HIGH_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.60
FUZZY_CUTOFF = 0.80

# Common OCR digit -> letter confusions, only used to propose candidates.
_OCR_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "0": ("o", "O"),
    "1": ("l", "I"),
    "2": ("Z",),
    "5": ("s", "S"),
    "6": ("G",),
    "8": ("B",),
}

TRUSTED_MEDICINES: set[str] = {
    "Amoxicillin",
    "Azithromycin",
    "Paracetamol",
    "Ibuprofen",
    "Metformin",
    "Atorvastatin",
    "Omeprazole",
    "Cetirizine",
    "Lisinopril",
    "Amlodipine",
}

OCR_FIXTURE: list[dict[str, Any]] = [
    {"id": "R1", "text": "Amoxicillin", "confidence": 0.95},
    {"id": "R2", "text": "Paracetamol", "confidence": 0.92},
    {"id": "R3", "text": "Ibuprofen", "confidence": 0.88},
    {"id": "R4", "text": "Metfromin", "confidence": 0.85},
    {"id": "R5", "text": "Am0xicillin", "confidence": 0.55},
    {"id": "R6", "text": "Cetirizin", "confidence": 0.72},
    {"id": "R7", "text": "Omeprazole", "confidence": 0.60},
    {"id": "R8", "text": "Aspirin", "confidence": 0.30},
]


@dataclass
class CorrectionRecord:
    """A single proposed correction; ``original`` is never overwritten."""

    original: str
    corrected: str
    reason: str
    confidence: float
    action: str  # "applied", "suggested" or "flagged"


def classify_confidence(confidence: float) -> str:
    """``>=0.90`` HIGH_CONFIDENCE, ``0.60..0.89`` REVIEW_REQUIRED, else LOW."""
    if confidence >= HIGH_THRESHOLD:
        return "HIGH_CONFIDENCE"
    if confidence >= REVIEW_THRESHOLD:
        return "REVIEW_REQUIRED"
    return "LOW_CONFIDENCE"


def clean_text(text: str) -> str:
    """Strip and collapse whitespace without changing any characters."""
    return re.sub(r"\s+", " ", text or "").strip()


def _confusion_candidates(name: str) -> Iterator[str]:
    """Yield single-character OCR repairs of ``name`` (suggestions only)."""
    for index, char in enumerate(name):
        for replacement in _OCR_CONFUSIONS.get(char, ()):
            yield name[:index] + replacement + name[index + 1 :]


def validate_medicine(
    name: str, trusted_dict: Iterable[str] | None = None
) -> dict[str, Any]:
    """Validate a medicine name against a trusted dictionary.

    Returns ``{valid, matched, method, needs_review}``.  Exact matches are
    ``method="exact"``; fuzzy hits are ``method="fuzzy"`` and always require
    review; anything else is ``method="none"``.
    """
    trusted = set(TRUSTED_MEDICINES if trusted_dict is None else trusted_dict)
    lookup = {med.lower(): med for med in trusted}
    cleaned = clean_text(name)

    if not cleaned:
        return {"valid": False, "matched": None, "method": "none", "needs_review": True}

    exact = lookup.get(cleaned.lower())
    if exact is not None:
        return {"valid": True, "matched": exact, "method": "exact", "needs_review": False}

    close = difflib.get_close_matches(cleaned.lower(), list(lookup), n=1, cutoff=FUZZY_CUTOFF)
    if close:
        return {
            "valid": True,
            "matched": lookup[close[0]],
            "method": "fuzzy",
            "needs_review": True,
        }
    return {"valid": False, "matched": None, "method": "none", "needs_review": True}


def _build_correction(
    original: str,
    cleaned: str,
    confidence: float,
    category: str,
    trusted: set[str],
) -> CorrectionRecord | None:
    """Decide a safe correction for one field, honouring the confidence gate."""
    result = validate_medicine(cleaned, trusted)
    if result["valid"] and result["method"] == "exact":
        return None  # already trustworthy, nothing to correct

    for candidate in _confusion_candidates(cleaned):
        repaired = validate_medicine(candidate, trusted)
        if repaired["valid"] and repaired["method"] == "exact":
            if category == "HIGH_CONFIDENCE":
                return CorrectionRecord(
                    original, candidate, "high-confidence OCR character correction",
                    confidence, "applied",
                )
            reason = (
                "medium confidence: character correction suggested for review"
                if category == "REVIEW_REQUIRED"
                else "low confidence: human review required before correcting"
            )
            return CorrectionRecord(original, candidate, reason, confidence, "suggested")

    if result["valid"] and result["method"] == "fuzzy":
        reason = (
            "fuzzy match to trusted medicine; confirm before applying"
            if category != "LOW_CONFIDENCE"
            else "low confidence: human review required before correcting"
        )
        return CorrectionRecord(original, result["matched"], reason, confidence, "suggested")

    if category == "LOW_CONFIDENCE":
        return CorrectionRecord(
            original, cleaned, "low confidence: no safe correction, flagged for human",
            confidence, "flagged",
        )
    return None


def review_ocr_results(
    results: Sequence[dict[str, Any]],
    medicine_dict: Iterable[str] | None = None,
    corrections: list[CorrectionRecord] | None = None,
) -> list[dict[str, Any]]:
    """Review OCR results, returning ``{original, cleaned, category, correction}``.

    ``corrections`` is an optional store; every emitted ``CorrectionRecord`` is
    appended to it so the original text stays auditable.  ``correction`` in the
    output is ``None`` or ``{corrected_text, reason, applied}``.
    """
    trusted = set(TRUSTED_MEDICINES if medicine_dict is None else medicine_dict)
    output: list[dict[str, Any]] = []

    for item in results:
        original = item.get("text", "")
        confidence = item.get("confidence", 0.0)
        cleaned = clean_text(original)
        category = classify_confidence(confidence)

        correction = _build_correction(original, cleaned, confidence, category, trusted)
        if correction is not None and corrections is not None:
            corrections.append(correction)

        correction_out = (
            {
                "corrected_text": correction.corrected,
                "reason": correction.reason,
                "applied": correction.action == "applied",
            }
            if correction is not None
            else None
        )
        output.append(
            {
                "original": original,
                "cleaned": cleaned,
                "category": category,
                "correction": correction_out,
            }
        )
    return output


if __name__ == "__main__":  # pragma: no cover - demo output
    store: list[CorrectionRecord] = []
    for row in review_ocr_results(OCR_FIXTURE, corrections=store):
        suggestion = row["correction"] or {}
        corrected = suggestion.get("corrected_text")
        applied = suggestion.get("applied")
        print(
            f"{row['original']!r} -> {row['category']} "
            f"(cleaned={row['cleaned']!r}, corrected={corrected}, applied={applied})"
        )
    print("Corrections recorded:", len(store))
