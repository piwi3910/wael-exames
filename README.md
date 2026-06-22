# wael-exames — local exam grading

Grades scanned NESA exam PDFs on the DGX Spark using a two-stage local-LLM pipeline:
`pdftoppm` render → `qwen3-vl` (handwriting transcription, port 8003) →
`qwen3.6-35b` (grading, port 8888) → JSON + Markdown report.

## Run

    uv run python grade.py "Math paper.pdf" --subject Math --out out

Outputs `out/<stem>.transcript.json`, `out/<stem>.results.json`, `out/<stem>.report.md`.

## Test

    uv run pytest

Unit tests mock the LLM calls; no DGX needed. The live run hits the DGX endpoints in
`examgrader/config.py`.

## How it works

1. `examgrader/pdf_to_images.py` — `pdftoppm` renders pages; near-blank scans are dropped.
2. `examgrader/transcriber.py` — sends each page to the vision model, which returns
   structured `{question, max_marks, student_answer, read_confidence}` records.
3. `examgrader/grader.py` — the `MarkScheme` interface grades each question. The POC uses
   `LLMJudge` (the reasoning model decides correctness). Production swaps in a
   marking-guide implementation without touching the reading stage.
4. `examgrader/report.py` — writes the per-question JSON + a readable Markdown report.

## Verified results (2026-06-22)

- Math paper: 68/100. Transcription of the objective section matched ground truth exactly
  (Q1 a–e, Q2, Q3a).
- English paper: 94.5/100.

## Known limitations (POC)

- **Section-header noise (English):** the vision model sometimes transcribes the
  section-overview lines ("Section A: Comprehension (20 marks)") as empty pseudo-questions.
  They score 0 and are flagged `low_read_confidence`, so they don't inflate the score, but
  they add noise to the report. Tightening `TRANSCRIBE_PROMPT` to ignore instruction/summary
  lines is the obvious follow-up.
- **`max_total` is fixed at 100** rather than derived from the transcribed questions.
- LLM-judge grading is best-effort; production should use the official marking guide via a
  `MarkScheme` implementation.

## Notes

- Endpoints, model names, and render DPI live in `examgrader/config.py`.
- Student PDFs and rendered pages are gitignored (they contain minors' names).
- DGX serving details (the `qwen3-vl` container, memory tuning) are in
  `docs/superpowers/specs/2026-06-22-exam-grading-framework-design.md`.
