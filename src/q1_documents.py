"""Question 1 — document de-duplication and quality classification.

Duplicate strategy
------------------
Several scans can share the same document ``id``.  We keep the record with the
**highest quality** score because a later re-capture of the same page is almost
always an attempt to repair a poor scan, so the best available quality is the
strongest evidence of content fidelity.  Ties on quality are broken by the
**earliest timestamp**, which makes the outcome fully deterministic and
reproducible.  Every discarded record is written to the logger with its id and
the reason, so the decision is auditable.  The caller's ``documents`` list is
never mutated: all work happens on a ``copy.deepcopy`` of the input.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

ACCEPT_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.50

VALID_STRATEGIES = ("keep_best", "keep_first")


def classify_quality(quality: float) -> str:
    """Classify a quality score.

    ``>= 0.80`` -> ``ACCEPT``; ``0.50 <= q < 0.80`` -> ``REVIEW``;
    ``< 0.50`` -> ``RETAKE``.
    """
    if quality >= ACCEPT_THRESHOLD:
        return "ACCEPT"
    if quality >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "RETAKE"


def deduplicate_documents(
    documents: Iterable[dict[str, Any]], strategy: str = "keep_best"
) -> list[dict[str, Any]]:
    """Return one record per document id, without mutating ``documents``.

    ``strategy="keep_best"`` (default) keeps the highest quality record and
    breaks ties on the earliest ``time``.  ``strategy="keep_first"`` keeps the
    first record seen for each id.  Output preserves first-appearance order.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of {VALID_STRATEGIES}"
        )

    working = copy.deepcopy(list(documents))
    best: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []

    for record in working:
        doc_id = record.get("id")
        if doc_id not in best:
            best[doc_id] = record
            order.append(doc_id)
            continue

        if strategy == "keep_first":
            logger.info("duplicate removed: id=%s (keep_first)", doc_id)
            continue

        current = best[doc_id]
        new_quality = record.get("quality", float("-inf"))
        cur_quality = current.get("quality", float("-inf"))
        new_time = record.get("time", float("inf"))
        cur_time = current.get("time", float("inf"))

        if new_quality > cur_quality or (
            new_quality == cur_quality and new_time < cur_time
        ):
            logger.info(
                "duplicate removed: id=%s (replaced: quality=%s time=%s)",
                doc_id,
                new_quality,
                new_time,
            )
            best[doc_id] = record
        else:
            logger.info("duplicate removed: id=%s (kept existing)", doc_id)

    return [best[doc_id] for doc_id in order]


def process_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Deduplicate, classify and summarise ``documents``.

    Returns a dict with ``unique_documents``, ``classified`` (each unique record
    augmented with a ``status``), per-category ``counts``, ``average_quality``
    (``0.0`` for an empty input) and ``duplicates_removed``.
    """
    unique_documents = deduplicate_documents(documents, strategy="keep_best")

    classified: list[dict[str, Any]] = []
    counts = {"ACCEPT": 0, "REVIEW": 0, "RETAKE": 0}
    total_quality = 0.0

    for record in unique_documents:
        status = classify_quality(record.get("quality", 0.0))
        counts[status] += 1
        total_quality += record.get("quality", 0.0)
        enriched = dict(record)
        enriched["status"] = status
        classified.append(enriched)

    average_quality = total_quality / len(unique_documents) if unique_documents else 0.0

    return {
        "unique_documents": unique_documents,
        "classified": classified,
        "counts": counts,
        "average_quality": average_quality,
        "duplicates_removed": len(list(documents)) - len(unique_documents),
    }


if __name__ == "__main__":  # pragma: no cover - demo output
    sample_documents = [
        {"id": "D101", "quality": 0.92, "time": 101},
        {"id": "D102", "quality": 0.43, "time": 102},
        {"id": "D103", "quality": 0.81, "time": 103},
        {"id": "D101", "quality": 0.92, "time": 104},
        {"id": "D104", "quality": 0.28, "time": 105},
        {"id": "D105", "quality": 0.74, "time": 105},
        {"id": "D102", "quality": 0.92, "time": 106},
    ]

    result = process_documents(sample_documents)
    print("Unique documents:", len(result["unique_documents"]))
    for record in result["classified"]:
        print(f"  {record['id']}: quality={record['quality']} -> {record['status']}")
    print("Counts:", result["counts"])
    print(f"Average quality: {result['average_quality']:.3f}")
    print("Duplicates removed:", result["duplicates_removed"])
