"""Tests for Question 5 - image -> structured pipeline (offline, deterministic).

Run: ``python3 -m pytest tests/test_q5.py -v``
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from q5_pipeline import TextExtractionPipeline  # noqa: E402

EXPECTED_TEXT = "\n".join(
    [
        "Patient: Rahul Sharma",
        "Medicine: Paracetamol",
        "Strength: 500 mg",
        "Frequency: Twice daily",
        "Duration: 5 days",
    ]
)


def _clear_image(seed: int = 7) -> np.ndarray:
    """Seeded white board with sharp cv2.putText lines (+ tiny seeded noise)."""
    rng = np.random.default_rng(seed)
    img = np.full((240, 640, 3), 255, np.uint8)
    for i, line in enumerate(EXPECTED_TEXT.splitlines()):
        cv2.putText(img, line, (20, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _bad_image(seed: int = 11) -> np.ndarray:
    """Seeded heavily blurred + underexposed image (unusable for OCR)."""
    blurred = cv2.GaussianBlur(_clear_image(seed), (41, 41), 0)
    return cv2.convertScaleAbs(blurred, alpha=0.3, beta=0)


def test_clear_image_extracts_text_and_calls_llm() -> None:
    """(a) good image -> structured fields + LLM path taken + cost recorded."""
    pipeline = TextExtractionPipeline()
    record, metrics = pipeline.run(_clear_image(), expected_text=EXPECTED_TEXT)
    print("\n[clear] record:", record)
    print("[clear] metrics:", metrics)

    assert record["validation_status"] in ("VALID", "REVIEW_REQUIRED")
    assert record["medicine_name"] == "Paracetamol"
    assert record["strength"] == "500 mg"
    assert record["duration"] == "5 days"
    assert metrics["llm_calls"] >= 1
    assert metrics["cost_usd"] > 0.0
    assert metrics["quality"]["is_usable"] is True
    assert metrics["latency_ms"] > 0.0


def test_bad_image_short_circuits_without_llm() -> None:
    """(b) unusable image -> NEEDS_RETAKE/REVIEW and zero LLM calls/cost."""
    pipeline = TextExtractionPipeline()
    record, metrics = pipeline.run(_bad_image())
    print("\n[bad] record:", record)
    print("[bad] metrics:", metrics)

    assert record["validation_status"] in ("NEEDS_RETAKE", "REVIEW_REQUIRED")
    assert metrics["llm_calls"] == 0
    assert metrics["cost_usd"] == 0.0
    assert metrics["quality"]["is_usable"] is False
    assert record["medicine_name"] == ""


def test_cache_hit_on_repeat_run() -> None:
    """(c) repeat run is served from cache: 0 LLM calls, cheaper + faster."""
    pipeline = TextExtractionPipeline()
    image = _clear_image()
    _, first = pipeline.run(image, expected_text=EXPECTED_TEXT)
    _, second = pipeline.run(image, expected_text=EXPECTED_TEXT)
    print("\n[first] metrics:", first)
    print("[second] metrics:", second)

    assert first["llm_calls"] == 1
    assert first["cached"] is False
    assert second["llm_calls"] == 0
    assert second["cost_usd"] == 0.0
    assert second["cached"] is True
    assert second["latency_ms"] < first["latency_ms"]
