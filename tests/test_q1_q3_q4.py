"""Tests for Questions 1, 3 and 4 — plain asserts, no third-party deps."""

import copy

from src.q1_documents import (
    classify_quality,
    deduplicate_documents,
    process_documents,
)
from src.q3_ocr import (
    CorrectionRecord,
    OCR_FIXTURE,
    classify_confidence,
    clean_text,
    review_ocr_results,
    validate_medicine,
)
from src.q4_structured import (
    build_prescription_json,
    parse_prescription_text,
    validate_prescription,
)

# ---------------------------------------------------------------------------
# Q1 fixtures
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {"id": "D101", "quality": 0.92, "time": 101},
    {"id": "D102", "quality": 0.43, "time": 102},
    {"id": "D103", "quality": 0.81, "time": 103},
    {"id": "D101", "quality": 0.92, "time": 104},
    {"id": "D104", "quality": 0.28, "time": 105},
    {"id": "D105", "quality": 0.74, "time": 105},
    {"id": "D102", "quality": 0.92, "time": 106},
]


def test_q1_counts_and_average():
    result = process_documents(DOCUMENTS)
    assert len(result["unique_documents"]) == 5
    assert result["duplicates_removed"] == 2
    assert result["counts"] == {"ACCEPT": 3, "REVIEW": 1, "RETAKE": 1}
    assert abs(result["average_quality"] - 0.734) < 1e-9


def test_q1_keep_best_selection():
    unique = deduplicate_documents(DOCUMENTS, strategy="keep_best")
    by_id = {doc["id"]: doc for doc in unique}
    assert by_id["D101"]["quality"] == 0.92 and by_id["D101"]["time"] == 101
    assert by_id["D102"]["quality"] == 0.92 and by_id["D102"]["time"] == 106
    assert by_id["D103"]["quality"] == 0.81
    assert by_id["D104"]["quality"] == 0.28
    assert by_id["D105"]["quality"] == 0.74


def test_q1_no_mutation():
    snapshot = copy.deepcopy(DOCUMENTS)
    process_documents(DOCUMENTS)
    assert DOCUMENTS == snapshot
    assert all("status" not in doc for doc in DOCUMENTS)


def test_q1_empty_input():
    result = process_documents([])
    assert result["unique_documents"] == []
    assert result["counts"] == {"ACCEPT": 0, "REVIEW": 0, "RETAKE": 0}
    assert result["average_quality"] == 0.0
    assert result["duplicates_removed"] == 0


def test_q1_classify_boundaries():
    assert classify_quality(0.80) == "ACCEPT"
    assert classify_quality(0.50) == "REVIEW"
    assert classify_quality(0.79) == "REVIEW"
    assert classify_quality(0.49) == "RETAKE"


# ---------------------------------------------------------------------------
# Q3 tests
# ---------------------------------------------------------------------------

def test_q3_confidence_boundaries():
    assert classify_confidence(0.90) == "HIGH_CONFIDENCE"
    assert classify_confidence(0.60) == "REVIEW_REQUIRED"
    assert classify_confidence(0.59) == "LOW_CONFIDENCE"


def test_q3_clean_text_normalises_only():
    assert clean_text("  Amoxicillin   500  mg  ") == "Amoxicillin 500 mg"
    assert clean_text("") == ""


def test_q3_correction_store_preserves_original():
    store: list[CorrectionRecord] = []
    review_ocr_results(OCR_FIXTURE, corrections=store)
    assert len(store) >= 1
    assert store[0].original == "Metfromin"  # original never overwritten
    assert all(rec.original != rec.corrected for rec in store if rec.action == "applied")


def test_q3_high_confidence_applies_repair():
    store: list[CorrectionRecord] = []
    rows = review_ocr_results(
        [{"id": "X", "text": "Am0xicillin", "confidence": 0.95}], corrections=store
    )
    assert rows[0]["category"] == "HIGH_CONFIDENCE"
    assert rows[0]["correction"]["corrected_text"] == "Amoxicillin"
    assert rows[0]["correction"]["applied"] is True


def test_q3_low_confidence_never_auto_corrects():
    store: list[CorrectionRecord] = []
    rows = review_ocr_results(
        [{"id": "X", "text": "Am0xicillin", "confidence": 0.55}], corrections=store
    )
    assert rows[0]["category"] == "LOW_CONFIDENCE"
    assert rows[0]["correction"]["applied"] is False
    assert "human" in rows[0]["correction"]["reason"].lower()


def test_q3_medicine_fuzzy_match():
    result = validate_medicine("Metfromin")
    assert result["valid"] is True
    assert result["matched"] == "Metformin"
    assert result["method"] == "fuzzy"
    assert result["needs_review"] is True


def test_q3_medicine_exact_match_case_insensitive():
    result = validate_medicine("amoxicillin")
    assert result["valid"] is True
    assert result["matched"] == "Amoxicillin"
    assert result["method"] == "exact"
    assert result["needs_review"] is False


# ---------------------------------------------------------------------------
# Q4 tests
# ---------------------------------------------------------------------------

VALID_TEXT = (
    "Patient: John Doe\n"
    "Medicine: Amoxicillin\n"
    "Strength: 500 mg\n"
    "Frequency: twice daily\n"
    "Duration: 7 days\n"
)


def test_q4_valid_example():
    result = build_prescription_json(VALID_TEXT)
    assert result["validation_status"] == "VALID"
    assert result["patient_name"] == "John Doe"
    assert result["medicine_name"] == "Amoxicillin"
    assert result["issues"] == []


def test_q4_missing_medicine_is_invalid():
    text = "Patient: Sam Poe\nStrength: 10 mg\nFrequency: once daily\nDuration: 5 days\n"
    result = build_prescription_json(text)
    assert result["validation_status"] == "INVALID"
    assert any("medicine" in issue for issue in result["issues"])


def test_q4_swapped_strength_frequency_flagged():
    text = (
        "Patient: Jane Roe\n"
        "Medicine: Ibuprofen\n"
        "Strength: twice daily\n"
        "Frequency: 400 mg\n"
        "Duration: 5 days\n"
    )
    result = build_prescription_json(text)
    assert result["validation_status"] == "REVIEW_REQUIRED"
    assert any("swapped" in issue for issue in result["issues"])


def test_q4_bad_duration_flagged():
    text = (
        "Patient: John Doe\n"
        "Medicine: Amoxicillin\n"
        "Strength: 500 mg\n"
        "Frequency: twice daily\n"
        "Duration: sometime\n"
    )
    result = build_prescription_json(text)
    assert result["validation_status"] == "REVIEW_REQUIRED"
    assert any("duration" in issue for issue in result["issues"])


def test_q4_parse_case_insensitive_keys():
    record = parse_prescription_text(
        "PATIENT: Ada\nMEDICINE: Metformin\nSTRENGTH: 850 mg\n"
        "FREQUENCY: once daily\nDURATION: 30 days\n"
    )
    assert record["patient_name"] == "Ada"
    assert record["medicine_name"] == "Metformin"
    assert validate_prescription(record)["status"] == "VALID"