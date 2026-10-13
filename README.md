# Python Intern Interview L1 — Complete Solution (Q1–Q7)

## Single-file submission
**`PythonInternInterview-L1.ipynb` is the one file containing all answers (Q1–Q7)
with executed outputs** — fully self-contained: each question's complete
implementation is inline in the notebook (zero imports from `src/`), followed by
its demo output plus a final verification cell with inline assertions covering
all 7 questions. To run on Google Colab / Kaggle: open the `.ipynb` alone and
**Runtime → Run all** (first cell installs opencv/numpy/pillow).

The repo additionally keeps the modular layout for review: `src/` sources plus
the formal `tests/` suite (`python -m pytest tests/ -v`, 48 tests).

## Setup (fresh runtime, local)
```bash
pip install opencv-python-headless numpy pillow pytest nbformat nbclient ipykernel
python -m pytest tests/ -v          # 48 tests, all offline / deterministic
```
Or open `PythonInternInterview-L1.ipynb` (Google Colab / Kaggle / Jupyter) and **Run All** —
it installs its own dependencies in the first cell and runs Q1→Q7 end-to-end.

## Repo layout
```
src/
  q1_documents.py       voice-record dedup (keep-best) + ACCEPT/REVIEW/RETAKE + average
  q2_image_quality.py   blur (Laplacian variance) + brightness/contrast gate, synthetic fixtures
  q3_ocr.py             OCR confidence triage + whitespace-safe cleaning + correction store + medicine dict
  q4_structured.py      prescription text → validated JSON (VALID/REVIEW_REQUIRED/INVALID, never guesses)
  q5_pipeline.py        image → structured pipeline: preprocess, gate, route, extract, validate, cache, JSON
  q6_chatbot.py         multi-KB chatbot: ACL, isolated retrieval, citations, history, topic switch, guardrails
  q7_legal_pipeline.py  legal STT-sim → correction → relevance filter → versioned-DB extraction (EN/HI/Hinglish)
tests/                  pytest suite, 48 tests (no network, no model downloads)
samples/  sample_outputs/  docs/      supporting artefacts
PythonInternInterview-L1.ipynb       executed notebook with all outputs
RESULTS_REPORT.md                    results, thresholds, AI-usage disclosure
```

## Design decisions (one line each)
- **Q1** — duplicates keep highest quality (re-capture = repair attempt), tie → earliest time; auditable log, input never mutated.
- **Q2** — Laplacian variance ≥ 100 + brightness in (60, 200); junk rejected before any OCR/LLM spend.
- **Q3** — confidence-gated corrections (<0.60 never auto-corrected); originals preserved in `CorrectionRecord`; medicines validated exact-then-fuzzy (cutoff 0.80).
- **Q4** — regex validation for dose/duration + swap detection; missing medicine → INVALID, never guessed.
- **Q5** — local-first: quality gate short-circuits junk at $0 LLM cost; result cache keys on image hash; deterministic offline stub extractor.
- **Q6** — retriever scoped to exactly one project (leak-proof by construction); token-overlap scoring, no embeddings needed.
- **Q7** — versioned local legal DB; unlisted sections return `{known: false}` instead of invented references; negations preserved.
# python-intern-interview-l1
