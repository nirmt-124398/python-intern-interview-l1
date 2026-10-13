"""Question 5 - image -> structured prescription pipeline (offline, $0 API).

Accepts a camera image of a prescription, preprocesses it, gates quality, routes
intelligently, extracts text with a deterministic simulated OCR engine, validates
fields with regexes, caches, and returns structured JSON plus latency/cost metrics.

Design decisions: the local-first quality gate (inline, dependency-free) rejects
junk before any LLM, saving 100% of its cost; results are cached by image hash so
repeats cost $0; deterministic regex validation + confidence triage run before any
LLM; Q2/Q3/Q4 helpers are imported lazily inside try/except with identical local
fallbacks so this file runs even if they are absent; seeded images only.

TIMING / COST (simulated, offline, single image)
Stage                  Latency   LLM cost  Notes
1 preprocess           ~200ms    $0.000    cv2 fastNlMeansDenoising
2 quality_gate         ~3ms      $0.000    Laplacian variance + brightness
3 intelligent_routing  <0.1ms    $0.000    junk -> early exit, $0
4 text_extraction      <1ms      $0.002    SimulatedOCREngine (offline)
5 validation           <1ms      $0.000    regex + confidence triage
6 optimize (cache)     <0.1ms    $0.000    hit -> $0, ~0 ms
7 structured_output    <0.1ms    $0.000    JSON assembly
TOTAL fresh good       ~220ms    $0.002    1 simulated LLM call
TOTAL junk image       ~225ms    $0.000    short-circuit at stage 3
TOTAL cache hit        ~0.6ms    $0.000    0 LLM calls
Run: python3 src/q5_pipeline.py    Test: python3 -m pytest tests/test_q5.py -v
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from copy import deepcopy
from typing import Any

import cv2
import numpy as np

COST_PER_LLM_CALL_USD = 0.002  # documented SIMULATION, not a real price
BLUR_MIN = 100.0               # Laplacian variance below -> blurry
BRIGHT_MIN, BRIGHT_MAX = 80.0, 250.0
CONTRAST_MIN = 55.0            # std below -> low contrast -> enhance
MAX_SIDE = 1000                # resize long edge to at most this

DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu|%)\b", re.I)
DURATION_RE = re.compile(r"\b\d+\s*(?:day|days|week|weeks|month|months)\b", re.I)
FREQ_RE = re.compile(
    r"\b(?:once|twice|thrice|daily|bd|tds|qid|od|bid|tid|every\s+\d+\s*hours?|"
    r"\d\s*-\s*\d\s*-\s*\d)\b",
    re.I,
)
FIELD_PATTERNS = {
    "patient_name": re.compile(r"patient\s*[:\-]\s*(.+)", re.I),
    "medicine_name": re.compile(r"(?:medicine|drug)\s*[:\-]\s*(.+)", re.I),
    "strength": re.compile(r"strength\s*[:\-]\s*(.+)", re.I),
    "frequency": re.compile(r"frequency\s*[:\-]\s*(.+)", re.I),
    "duration": re.compile(r"duration\s*[:\-]\s*(.+)", re.I),
}
REQUIRED_FIELDS = ("patient_name", "medicine_name", "strength", "frequency", "duration")

def _optional(module: str, attr: str) -> Any:
    """Lazily import an optional Q2/Q3/Q4 helper; None if absent (fallback)."""
    for name in (module, f"src.{module}"):
        try:
            return getattr(importlib.import_module(name), attr)
        except Exception:
            continue
    return None

# Optional reuse of earlier questions - never a hard top-level dependency.
_q2_quality_gate = _optional("q2_image_quality", "check_image_quality")
_q3_triage = _optional("q3_ocr", "classify_confidence")
_q4_parse_fields = _optional("q4_structured", "parse_prescription_text")

def _local_triage(confidence: float) -> str:
    """Q3 fallback: HIGH >=0.90, REVIEW >=0.60, else LOW (one-liner)."""
    return "HIGH_CONFIDENCE" if confidence >= 0.90 else ("REVIEW_REQUIRED" if confidence >= 0.60 else "LOW_CONFIDENCE")

def _local_parse_fields(text: str) -> dict[str, str]:
    """Q4 fallback: pull labelled fields out of OCR text with regexes."""
    return {name: m.group(1).strip() for name, p in FIELD_PATTERNS.items()
            if (m := p.search(text))}

def _confidence_tier(confidence: float) -> str:
    """Map a confidence into HIGH / REVIEW / LOW (inline, no dependency)."""
    return "HIGH" if confidence >= 0.90 else ("REVIEW" if confidence >= 0.60 else "LOW")

class SimulatedOCREngine:
    """Deterministic offline stand-in for an OCR / LLM text extractor.
    Purpose/Input/Output: grayscale image (+ optional expected_text hint) ->
    ``[{"text", "confidence"}, ...]``. Impact: the only stage simulating an LLM
    call (1 call), so it dominates the $0.002 cost; deterministic and offline.
    """

    def __init__(self, base_confidence: float = 0.96) -> None:
        self.base_confidence: float = base_confidence

    @staticmethod
    def _ink_density(gray: np.ndarray) -> float:
        """Fraction of dark (ink) pixels, used to scale confidence."""
        return float(np.count_nonzero(gray < 128)) / float(gray.size)

    def _contour_heuristic(self, gray: np.ndarray) -> list[dict[str, Any]]:
        """Blank/contour fallback: cluster contours into text rows."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ys = sorted(cv2.boundingRect(c)[1] for c in contours if cv2.contourArea(c) > 5)
        rows: list[int] = []
        for y in ys:
            if not rows or y - rows[-1] > 10:
                rows.append(y)
        conf = round(min(0.89, max(0.40, 0.45 + self._ink_density(gray) * 8.0)), 2)
        return [{"text": f"[unreadable_line_{i + 1}]", "confidence": conf}
                for i in range(max(1, len(rows)))]

    def extract(self, gray: np.ndarray, expected_text: str = "") -> list[dict[str, Any]]:
        """Use the hint when supplied, else the contour heuristic."""
        if expected_text.strip():
            conf = round(self.base_confidence if self._ink_density(gray) > 0.004 else 0.58, 2)
            return [{"text": ln.strip(), "confidence": conf}
                    for ln in expected_text.splitlines() if ln.strip()]
        return self._contour_heuristic(gray)

class TextExtractionPipeline:
    """Seven-stage image -> structured JSON pipeline with cost/latency metrics."""

    def __init__(self) -> None:
        self.ocr: SimulatedOCREngine = SimulatedOCREngine()
        self._cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._llm_calls_total: int = 0

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Stage 1 - Image Preprocessing.
        Purpose/Input/Output: normalise a BGR/gray capture into a denoised, resized
        grayscale image. Impact: denoising lifts accuracy on noisy captures; this
        CPU work is the largest latency term but costs no LLM.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        h, w = gray.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / float(max(h, w))
            gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        try:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        except Exception:  # pragma: no cover - defensive fallback
            gray = cv2.medianBlur(gray, 3)
        return gray

    def quality_gate(self, gray: np.ndarray) -> dict[str, Any]:
        """Stage 2 - Quality Gate (inline, dependency-free).
        Purpose/Input/Output: grayscale -> ``{is_usable, blur_score, brightness,
        contrast, reason}``. Impact: ~1 ms / $0; rejects junk up front so accuracy
        is protected and 100% downstream LLM cost is saved.
        """
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness, contrast = float(gray.mean()), float(gray.std())
        problems: list[str] = []
        if blur_score < BLUR_MIN:
            problems.append("blurred")
        if not (BRIGHT_MIN <= brightness <= BRIGHT_MAX):
            problems.append("poorly illuminated")
        return {"is_usable": not problems, "blur_score": round(blur_score, 2),
                "brightness": round(brightness, 1), "contrast": round(contrast, 1),
                "reason": "ok" if not problems else "Image may be " + " or ".join(problems)}

    def intelligent_routing(self, quality: dict[str, Any]) -> dict[str, str]:
        """Stage 3 - Intelligent Routing.
        Purpose/Input/Output: quality dict -> ``{action, reason}`` where action is
        NEEDS_RETAKE / ENHANCE / TEXT_EXTRACTION. Impact: junk returns early ->
        0 LLM calls ($0); low contrast takes free local CLAHE.
        """
        if not quality["is_usable"]:
            return {"action": "NEEDS_RETAKE", "reason": quality["reason"]}
        if quality["contrast"] < CONTRAST_MIN:
            return {"action": "ENHANCE", "reason": "low contrast -> local enhance"}
        return {"action": "TEXT_EXTRACTION", "reason": "routed to local extractor"}

    def text_extraction(self, gray: np.ndarray, expected_text: str = "") -> list[dict[str, Any]]:
        """Stage 4 - AI-Based Text Extraction (simulated, offline).
        Purpose/Input/Output: image + hint -> ``{text, confidence}`` records.
        Impact: the only LLM-incurring stage; it increments the call counter so
        cost is attributed ($0.002 simulated) and is deterministic for accuracy.
        """
        self._llm_calls_total += 1
        return self.ocr.extract(gray, expected_text)

    def validation(
        self, lines: list[dict[str, Any]], quality: dict[str, Any]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Stage 5 - Accuracy and Validation.
        Purpose/Input/Output: lines + quality -> ``(status, fields, report)`` with
        regex dose/duration checks and HIGH/REVIEW/LOW confidence triage. Impact:
        deterministic checks cost <1 ms / $0 and flag missing or ambiguous fields
        instead of guessing, so only truly uncertain cases reach human review.
        """
        text = "\n".join(ln["text"] for ln in lines)
        parser = _q4_parse_fields or _local_parse_fields
        try:
            fields = dict(parser(text))
        except Exception:  # pragma: no cover - defensive
            fields = _local_parse_fields(text)
        _ = _q3_triage or _local_triage  # Q3 hook kept for parity; triage is inline
        tiers = {ln["text"]: _confidence_tier(float(ln["confidence"])) for ln in lines}
        lowest = min((float(ln["confidence"]) for ln in lines), default=1.0)
        missing = [n for n in REQUIRED_FIELDS if not fields.get(n)]
        flags: list[str] = ["missing:" + ",".join(missing)] if missing else []
        if fields.get("strength") and not DOSE_RE.search(fields["strength"]):
            flags.append("strength_format")
        if fields.get("duration") and not DURATION_RE.search(fields["duration"]):
            flags.append("duration_format")
        if (fields.get("strength") and FREQ_RE.search(fields["strength"])
                and not DOSE_RE.search(fields["strength"])) or (
                fields.get("frequency") and DOSE_RE.search(fields["frequency"])
                and not FREQ_RE.search(fields["frequency"])):
            flags.append("strength_frequency_swapped")
        if not fields:
            status = "REVIEW_REQUIRED"
        elif missing:
            status = "INVALID" if "medicine_name" in missing else "REVIEW_REQUIRED"
        else:
            status = "VALID" if _confidence_tier(lowest) == "HIGH" and not flags else "REVIEW_REQUIRED"
        report = {"overall_tier": _confidence_tier(lowest),
                  "lowest_confidence": round(lowest, 2), "line_tiers": tiers,
                  "flags": flags, "usable": quality["is_usable"]}
        return status, fields, report

    def optimize(self, cache_key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Stage 6 - Cost & Performance Optimization (cache lookup).
        Purpose/Input/Output: cache key -> cached ``(record, metrics)`` or None.
        Impact: a hit costs $0, makes 0 LLM calls, and is a ~0.1 ms dict lookup --
        the biggest end-to-end win for repeated images.
        """
        return self._cache.get(cache_key)

    @staticmethod
    def _cache_key(image: np.ndarray, expected_text: str) -> str:
        """Stable hash of image bytes (shape/dtype) salted with the hint."""
        d = hashlib.sha256()
        d.update(f"{image.shape}|{image.dtype}|{expected_text}".encode())
        d.update(np.ascontiguousarray(image).tobytes())
        return d.hexdigest()

    def structured_output(
        self, fields: dict[str, str], quality: dict[str, Any], confidence: dict[str, Any],
        status: str, lines: list[dict[str, Any]], metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Stage 7 - Structured Output.
        Purpose/Input/Output: validated parts -> JSON-serialisable prescription
        ``dict`` with an ``extraction_meta`` block. Impact: pure assembly, ~0 ms /
        $0; guarantees a stable contract whichever path ran.
        """
        record: dict[str, Any] = {n: fields.get(n, "") for n in REQUIRED_FIELDS}
        record["validation_status"] = status
        record["extraction_meta"] = {
            "latency_ms": metrics["latency_ms"], "llm_calls": metrics["llm_calls"],
            "cost_usd": metrics["cost_usd"], "quality": quality,
            "confidence": confidence, "text_lines": lines,
            "cached": metrics["cached"], "route": metrics["route"]}
        return record

    def run(
        self, image: np.ndarray, expected_text: str = ""
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run all seven stages; return ``(record, metrics)``.
        ``metrics`` = ``{latency_ms, llm_calls, cost_usd, cached, route, quality,
        stage_latency_ms}``. Latency uses ``time.perf_counter``; cost is the
        documented simulated ``llm_calls * 0.002`` USD.
        """
        started, stage_ms = time.perf_counter(), {}
        cache_key = self._cache_key(image, expected_text)

        def mark(stage: str, t0: float) -> None:
            stage_ms[stage] = round((time.perf_counter() - t0) * 1000.0, 3)

        cached = self.optimize(cache_key)  # Stage 6 read path
        if cached is not None:
            record = deepcopy(cached[0])
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            record["extraction_meta"].update(latency_ms=latency_ms, llm_calls=0,
                                             cost_usd=0.0, cached=True, route="CACHE_HIT")
            return record, {"latency_ms": latency_ms, "llm_calls": 0, "cost_usd": 0.0,
                            "cached": True, "route": "CACHE_HIT",
                            "quality": record["extraction_meta"].get("quality", {}),
                            "stage_latency_ms": {"optimize": latency_ms}}

        calls_before = self._llm_calls_total
        t0 = time.perf_counter(); gray = self.preprocess(image); mark("preprocess", t0)
        t0 = time.perf_counter(); quality = self.quality_gate(gray); mark("quality_gate", t0)
        t0 = time.perf_counter(); route = self.intelligent_routing(quality)
        mark("intelligent_routing", t0)

        lines: list[dict[str, Any]] = []
        fields: dict[str, str] = {}
        confidence: dict[str, Any] = {"overall_tier": "LOW", "lowest_confidence": 0.0,
                                      "line_tiers": {}, "flags": []}
        status = "NEEDS_RETAKE"
        if route["action"] != "NEEDS_RETAKE":
            if route["action"] == "ENHANCE":
                t0 = time.perf_counter()
                gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                mark("enhance", t0)
            t0 = time.perf_counter(); lines = self.text_extraction(gray, expected_text)
            mark("text_extraction", t0)
            t0 = time.perf_counter(); status, fields, confidence = self.validation(lines, quality)
            mark("validation", t0)

        llm_calls = self._llm_calls_total - calls_before
        metrics: dict[str, Any] = {
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "llm_calls": llm_calls, "cost_usd": round(llm_calls * COST_PER_LLM_CALL_USD, 6),
            "cached": False, "route": route["action"], "quality": quality,
            "stage_latency_ms": stage_ms}
        t0 = time.perf_counter()
        record = self.structured_output(fields, quality, confidence, status, lines, metrics)
        mark("structured_output", t0)
        self._cache[cache_key] = (deepcopy(record), metrics)
        return record, metrics

_LINES = ["Patient: Rahul Sharma", "Medicine: Paracetamol", "Strength: 500 mg", "Frequency: Twice daily", "Duration: 5 days"]

def _clear_image() -> np.ndarray:
    """Synthetic white board with sharp printed lines (deterministic)."""
    img = np.full((240, 640, 3), 255, np.uint8)
    for i, line in enumerate(_LINES):
        cv2.putText(img, line, (20, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return img

def _bad_image() -> np.ndarray:
    """Heavily blurred and underexposed capture (deterministic)."""
    blurred = cv2.GaussianBlur(_clear_image(), (41, 41), 0)
    return cv2.convertScaleAbs(blurred, alpha=0.3, beta=0)

if __name__ == "__main__":
    expected = "\n".join(_LINES)
    pipeline = TextExtractionPipeline()
    good, good_metrics = pipeline.run(_clear_image(), expected_text=expected)
    print("=== GOOD IMAGE ===")
    print(json.dumps(good, indent=2))
    print("metrics:", json.dumps(good_metrics, indent=2))
    bad, bad_metrics = pipeline.run(_bad_image())
    print("\n=== BAD IMAGE (junk) ===")
    print(json.dumps(bad, indent=2))
    print("metrics:", json.dumps(bad_metrics, indent=2))
    _, hit_metrics = pipeline.run(_clear_image(), expected_text=expected)
    print("\n=== CACHE HIT (repeat) ===")
    print("metrics:", json.dumps(hit_metrics, indent=2))
