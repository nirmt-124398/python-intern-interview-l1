"""Question 4 — structured prescription extraction with schema validation.

LLM-extraction-with-schema-validation pattern
---------------------------------------------
1. Prompt the LLM to return a JSON object matching a fixed schema
   (``patient_name``, ``medicine_name``, ``strength``, ``frequency``,
   ``duration``).
2. ``json.loads`` the reply into a dict.
3. Run the *same* validator used for any other source on that dict.
4. Reject/repair loop: if the status is ``INVALID`` or ``REVIEW_REQUIRED`` the
   issues list is fed back to the model (or a human) for correction before the
   record is accepted.

The validator never guesses: missing or ambiguous fields produce an explicit
``issues`` entry and a ``REVIEW_REQUIRED`` status, and a missing medicine name
is ``INVALID`` outright.
"""

from __future__ import annotations

import re
from typing import Any

REQUIRED_FIELDS = ("patient_name", "medicine_name", "strength", "frequency", "duration")

_DURATION_RE = re.compile(r"^\d+\s*(day|days|week|weeks|month|months)$", re.IGNORECASE)
_DOSE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg/ml|mg|g|mcg|ml|tablets?|capsules?)", re.IGNORECASE
)
_FREQUENCY_KEYWORDS = (
    "daily",
    "once",
    "twice",
    "thrice",
    "od",
    "bd",
    "tds",
    "qid",
    "hour",
    "hours",
    "morning",
    "night",
    "weekly",
    "every",
)

_FIELD_ALIASES = {
    "patient": "patient_name",
    "patient name": "patient_name",
    "medicine": "medicine_name",
    "medicine name": "medicine_name",
    "drug": "medicine_name",
    "strength": "strength",
    "dose": "strength",
    "frequency": "frequency",
    "duration": "duration",
}


def parse_prescription_text(text: str) -> dict[str, str]:
    """Parse ``Key: value`` lines into a canonical prescription dict.

    Keys are matched case-insensitively; ``patient``/``medicine`` map to
    ``patient_name``/``medicine_name``.  Unrecognised lines are ignored.
    """
    record: dict[str, str] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        canonical = _FIELD_ALIASES.get(key.strip().lower())
        if canonical is not None:
            record[canonical] = value.strip()
    return record


def validate_prescription(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a parsed prescription without guessing.

    Returns ``{"status": ..., "issues": [...]}`` where status is ``VALID``
    (zero issues), ``INVALID`` (medicine name missing) or ``REVIEW_REQUIRED``.
    """
    issues: list[str] = []

    for field in REQUIRED_FIELDS:
        if not str(record.get(field) or "").strip():
            issues.append(f"missing required field: {field}")

    medicine = str(record.get("medicine_name") or "").strip()
    strength = str(record.get("strength") or "").strip()
    frequency = str(record.get("frequency") or "").strip()
    duration = str(record.get("duration") or "").strip()

    # Swap detection: strength holding a frequency keyword, or frequency
    # holding a dose pattern, means the two fields were crossed.
    if strength and frequency:
        strength_is_frequency = any(k in strength.lower() for k in _FREQUENCY_KEYWORDS)
        frequency_is_dose = _DOSE_PATTERN.search(frequency) is not None
        if strength_is_frequency or frequency_is_dose:
            issues.append("strength and frequency appear swapped")

    if duration and not _DURATION_RE.match(duration):
        issues.append(f"invalid or unreadable duration: {duration!r}")

    if strength and not _DOSE_PATTERN.search(strength):
        if not any(k in strength.lower() for k in _FREQUENCY_KEYWORDS):
            issues.append(f"strength does not contain a recognised dose: {strength!r}")

    if not medicine:
        status = "INVALID"
    elif issues:
        status = "REVIEW_REQUIRED"
    else:
        status = "VALID"
    return {"status": status, "issues": issues}


def build_prescription_json(text: str) -> dict[str, Any]:
    """Parse ``text`` and produce a schema-validated prescription record."""
    record = parse_prescription_text(text)
    result = validate_prescription(record)
    return {
        "patient_name": record.get("patient_name", ""),
        "medicine_name": record.get("medicine_name", ""),
        "strength": record.get("strength", ""),
        "frequency": record.get("frequency", ""),
        "duration": record.get("duration", ""),
        "validation_status": result["status"],
        "issues": result["issues"],
    }


if __name__ == "__main__":  # pragma: no cover - demo output
    valid_text = (
        "Patient: John Doe\n"
        "Medicine: Amoxicillin\n"
        "Strength: 500 mg\n"
        "Frequency: twice daily\n"
        "Duration: 7 days\n"
    )
    print("VALID example:", build_prescription_json(valid_text))

    swapped_text = (
        "Patient: Jane Roe\n"
        "Medicine: Ibuprofen\n"
        "Strength: twice daily\n"
        "Frequency: 400 mg\n"
        "Duration: 5 days\n"
    )
    print("SWAPPED example:", build_prescription_json(swapped_text))

    incomplete_text = "Patient: Sam Poe\nStrength: 10 mg\nFrequency: once daily\n"
    print("INCOMPLETE example:", build_prescription_json(incomplete_text))