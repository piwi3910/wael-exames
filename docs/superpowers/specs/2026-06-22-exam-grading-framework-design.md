# Local Exam Grading Framework — POC Design

**Date:** 2026-06-22
**Status:** Approved design (pre-implementation)

## Goal

Grade two scanned NESA Primary Leaving exam papers end-to-end on the DGX Spark,
producing a per-question breakdown plus total for each:

- `English paper.pdf` (Sections A–D: Comprehension, Vocabulary, Language use, Composition; /100)
- `Math paper.pdf` (numbered questions with "show your working"; /100)

Both are **scanned, completed** papers: printed questions with **handwritten student
answers** on noisy, low-contrast scans. Success bar for the POC is correct end-to-end
grading of these two specific papers (depth over breadth). One student per paper.

## Environment (verified 2026-06-22)

DGX Spark, vLLM, OpenAI-compatible endpoints:

- `http://192.168.10.246:8888/v1` → `qwen3.6-35b` (Qwen3.6-35B-A3B, **text-only**, 262k ctx)
- `http://192.168.10.246:8890/v1` → `bge-m3` (embeddings; not needed for POC)

**Gap:** no vision model is currently served. The inputs are images, so a vision model
must be added (see Vision Model below).

## Architecture

Two stages with one clean boundary: **Read** (vision) → **Judge** (reasoning). The
interface between them is a typed transcription object. This boundary is deliberate: in
production the judge swaps its LLM-decision logic for the official marking guide *without
touching the reading stage*.

```
PDF
 └─ pdf_to_images ──► [page PNGs]
       └─ transcriber (VLM) ──► TranscribedPaper
             └─ grader (qwen3.6-35b, LLM-judge) ──► GradedPaper
                   └─ report ──► results.json + report.md
```

### Mark scheme strategy

- **POC:** LLM-judge — the grader determines the correct answer from its own knowledge
  and awards marks out of the printed max ("(N marks)") with a justification.
- **Production:** official NESA marking guides will exist. The grader consumes a
  `MarkScheme` interface; an LLM-judge implementation is used now, a guide-driven
  implementation drops in later with no change to the reading stage.

## Components

Each component has one purpose, a typed interface, and is independently testable.

1. **`pdf_to_images`**
   - PyMuPDF renders each page to high-DPI PNG.
   - OpenCV light preprocessing (denoise, contrast, deskew) — the scans are speckly and
     low-contrast.
   - Skip near-blank pages.
   - Out: ordered list of page image paths.

2. **`transcriber`** (vision)
   - VLM client (OpenAI-compatible, multimodal `image_url`).
   - Per page image → structured records:
     `{section, question_no, max_marks, question_text, student_answer, read_confidence}`.
   - Persists transcripts to disk (enables re-grading without re-OCR; decouples the two
     models so they need not be co-resident).

3. **`grader`** (reasoning)
   - `qwen3.6-35b` client.
   - Consumes `TranscribedPaper` → awards marks per question with justification via the
     `MarkScheme` interface (LLM-judge implementation for POC).
   - Math: award method marks where working is shown.
   - Per-question isolation: one question's failure does not sink the paper.

4. **`report`**
   - Assembles per-question JSON (`results.json`) and a readable Markdown report
     (`report.md`): per-question extracted answer, awarded/max marks, justification,
     confidence; section subtotals; total /100.

5. **`config` / `cli`**
   - Endpoints, model names, prompts, DPI, paths.
   - `grade.py <pdf>` runs the full pipeline.

Typed Pydantic schemas define every cross-stage object (`PageImage`, `TranscribedQuestion`,
`TranscribedPaper`, `GradedQuestion`, `GradedPaper`).

## Vision Model (new dependency)

- **Recommended:** `Qwen2.5-VL-32B-Instruct` (AWQ/FP8) served via vLLM on a **new port
  (8889)** using the existing Spark Arena image — strong document + handwriting OCR and
  structured extraction.
- **Fallback:** `Qwen2.5-VL-7B-Instruct` if VRAM is tight alongside the 262k-context
  `qwen3.6-35b`.
- **First setup step:** verify VRAM coexistence on the DGX. Because transcripts are
  persisted, an acceptable fallback is to transcribe all pages first, then bring the judge
  up — the two models never strictly need to be co-resident.
- The exact serve command is produced during implementation setup.

## Data Flow & Schemas (interfaces)

- `PageImage { page_no, path, is_blank }`
- `TranscribedQuestion { section, question_no, max_marks, question_text, student_answer, read_confidence }`
- `TranscribedPaper { subject, source_pdf, questions: [TranscribedQuestion] }`
- `GradedQuestion { question_no, max_marks, awarded_marks, student_answer, justification, grade_confidence, flags }`
- `GradedPaper { subject, source_pdf, questions: [GradedQuestion], section_totals, total, max_total }`

## Error Handling

- JSON-parse retries on both VLM and grader calls (with guided-JSON where available).
- Low-confidence reads and grades flagged in the report for human attention.
- Persisted intermediate transcripts allow re-grading without re-OCR.
- Per-question try/except in the grader.

## Testing (TDD)

- Unit tests with **mocked LLM responses** for grader logic, report assembly, and schema
  validation, using a frozen golden-transcript fixture (deterministic).
- The actual 2-PDF run is a manual integration check (the live model cannot be unit
  tested); transcripts are snapshotted for reproducibility.

## Stack

Python, PyMuPDF, OpenCV, OpenAI-compatible client (pointed at the DGX vLLM endpoints),
Pydantic, pytest.

## Decisions / Defaults

- **Language:** Python (PyMuPDF + OpenCV are best-in-class and low-friction for the
  CV/PDF-heavy parts). Go was considered; rejected only because PDF rasterization in Go
  needs CGo (go-fitz) or a CLI (pdftoppm).
- **Per-question max marks** are read from the printed "(N marks)" labels.
- **Questions come from the scan itself** — no separate question bank, since only completed
  papers are available.

## Out of Scope (POC)

- Batch grading across many students.
- Teacher review/override UI.
- Production marking-guide ingestion (interface is provided; implementation deferred).
