"""Tests for the Question 7 local-first legal transcription pipeline."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.q7_legal_pipeline import (
    AUDIO_FIXTURES,
    DB_VERSION,
    IRRELEVANT,
    LEGALLY_RELEVANT,
    UNCERTAIN,
    SimulatedSTT,
    benchmark,
    classify_segment,
    correct_transcript,
    extract_legal_info,
    filter_transcript,
    run_pipeline,
)


def test_stt_fixture_loads() -> None:
    stt = SimulatedSTT()
    assert len(AUDIO_FIXTURES) == 3
    out = stt.transcribe("en_agreement_01")
    assert set(out) == {"transcript", "language", "wer_hint"}
    assert out["language"] == "en"
    assert isinstance(out["wer_hint"], float)
    assert "Section 999" in out["transcript"]


def test_correction_fixes_error_and_preserves_negation() -> None:
    fixed = correct_transcript("The defendant did not pay the amount twlce.")
    assert "twice" in fixed
    assert "twlce" not in fixed
    assert "not" in fixed

    section = correct_transcript("as per section seventy three we can claim")
    assert "Section 73" in section

    hindi = correct_transcript("भुगतान 15 March 2024 तक नहीं हुआ")
    assert "नहीं" in hindi


def test_relevance_labels_correct() -> None:
    assert classify_segment("The payment was not made as per the agreement.") == LEGALLY_RELEVANT
    assert classify_segment("The weather is nice and we ate food.") == IRRELEVANT
    assert classify_segment("He arrived at the office at noon.") == UNCERTAIN


def test_filter_keeps_relevant_and_uncertain_drops_irrelevant() -> None:
    segments = [
        "The agreement was signed.",
        "The weather is nice today.",
        "He arrived at noon.",
    ]
    kept, dropped = filter_transcript(segments)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0]["label"] == IRRELEVANT
    assert {item["label"] for item in kept} == {LEGALLY_RELEVANT, UNCERTAIN}


def test_extraction_finds_section_and_date() -> None:
    result = run_pipeline("en_agreement_01")
    extraction = result["extraction"]
    assert "Section 73" in extraction["sections_mentioned"]
    assert any("2023" in date for date in extraction["dates"])
    assert extraction["parties"]
    assert "Sharma versus Verma" in extraction["parties"]


def test_unknown_section_is_not_invented() -> None:
    extraction = extract_legal_info("The client mentioned Section 999 for cheating.")
    unknown = [p for p in extraction["suggested_provisions"] if p["section"] == "Section 999"]
    assert len(unknown) == 1
    assert unknown[0]["known"] is False
    assert "warning" in unknown[0]
    assert "not in local DB" in unknown[0]["warning"]
    assert "act" not in unknown[0]


def test_suggested_provision_flagged_not_stated() -> None:
    extraction = extract_legal_info("There was a payment default, clearly a breach.")
    suggested = [p for p in extraction["suggested_provisions"] if p["section"] == "Section 73"]
    assert suggested and suggested[0]["status"] == "SUGGESTED"
    assert "Section 73" not in extraction["sections_mentioned"]


def test_hinglish_fixture_processes() -> None:
    result = run_pipeline("hi_payment_01")
    assert result["language"] == "hinglish"
    assert "15 March 2024" in result["corrected_transcript"]
    assert any("2024" in date for date in result["extraction"]["dates"])
    assert any(seg["label"] == LEGALLY_RELEVANT for seg in result["segments"])
    assert len(result["relevant_only"]) >= 1


def test_benchmark_runs_with_zero_cost() -> None:
    rows = benchmark()
    assert len(rows) == 3
    assert all(row["api_cost_usd"] == 0.0 for row in rows)
    assert all(row["wer_proxy"] > 0 for row in rows)
    assert all(0.0 <= row["relevant_ratio"] <= 1.0 for row in rows)
    assert DB_VERSION == "1.0"
