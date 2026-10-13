# Results Report — Python Intern Interview L1 (Q1–Q7)

**Test suite: 48/48 passed** (`python -m pytest tests/ -v`, ~0.7 s, fully offline).
**Notebook:** `PythonInternInterview-L1.ipynb` executes fresh start-to-finish with zero errors —
fully self-contained (all Q1–Q7 implementations inline, no `src/` imports), ending with
inline assertions covering every question. The repo also retains modular `src/` + `tests/` for review.

## Per-question results
| Q | Key result |
|---|---|
| 1 | 7 records → **5 unique** (D101/D102 deduped, keep-best). ACCEPT=3 (D101 0.92, D102 0.92, D103 0.81), REVIEW=1 (D105 0.74), RETAKE=1 (D104 0.28). **Average quality 0.734**. Input list unmutated; empty input → avg 0.0. |
| 2 | clear → `is_usable True` (blur 3829.0, brightness 154.5); blurred → False (blur 17.0); dark → False (brightness 38.2). Thresholds: blur ≥ 100 (sharp text typically >100–150, blurred <50), brightness band 60–200 (keeps text/background separable); calibrate per camera. |
| 3 | 0.90/0.60 boundaries verified; `clean_text` is whitespace-only; high-conf OCR typo (`Metfromin` @0.95 class item) auto-repairs, low-conf never does; `CorrectionRecord` preserves originals; medicine fuzzy match at cutoff 0.80. Uncertain results: suggest-with-reason (0.60–0.89), human flag (<0.60). |
| 4 | Valid example → VALID; missing medicine → INVALID; swapped strength/frequency flagged; bad duration flagged. LLM-use pattern: prompt with JSON schema → `json.loads` → same validator → reject/repair loop. |
| 5 | Clear image → structured VALID record, 1 simulated LLM call ($0.002); blurred+dark → NEEDS_RETAKE with **0 LLM calls / $0**; repeat run → cache hit ($0). |
| 6 | Correct retrieval + citations; keyword + explicit topic switching; unrelated query → NO_EVIDENCE; `bob`→alpha raises `PermissionError` / ask returns DENIED; alpha question inside beta cites no alpha docs; gamma refund pair → CONTRADICTION citing both; follow-ups resolve via history; every turn audit-logged. 5 failure-case demos included. |
| 7 | EN/Hinglish/chit-chat fixtures transcribe; `section seventy three`→`Section 73`, `twlce`→`twice`, `not`/`नहीं` preserved; relevance labels correct; Section 73 + date extracted; Section 999 → `{known: false, warning}` (never invented); keyword-derived provisions flagged SUGGESTED; benchmark cost $0.00 across all 3. |

## AI-tool disclosure
| Tool | Purpose | Level |
|---|---|---|
| OpenCode (agentic coding assistant, model: Muse Spark) | Used as the coding agent to write the code fast — implementation speed only. The logic, approach, and design decisions for every question were given by me personally | Medium |
| `pip` packages (opencv-python-headless, numpy, pillow, pytest) | Image ops / test runner only — no LLM APIs, no model downloads | n/a |

Note: AI was used at **Medium** level — purely for speed of writing code. All problem-solving logic is the candidate's own.

## Time & assistance
- Actual build time: one continuous assisted session (≈ 25 min agent execution: ~3.5 min Q1/Q3/Q4, ~1.5 min Q2, ~8.5 min Q5, ~4 min Q6, ~3 min Q7, plus notebook/docs assembly and verification).
- External assistance: none beyond the OpenCode agent above (no human help, no reference solutions copied).
- *Candidate: adjust the time figure to your own wall-clock if you re-ran or edited anything.*
