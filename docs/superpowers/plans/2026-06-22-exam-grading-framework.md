# Exam Grading Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grade two scanned NESA exam PDFs end-to-end on the DGX — render pages, transcribe printed questions + handwritten answers with a vision model, grade with a reasoning model, and emit a per-question breakdown + total.

**Architecture:** Two-stage pipeline with a typed boundary. `qwen3-vl` (vision, port 8003) transcribes each page into structured questions/answers; `qwen3.6-35b` (reasoning, port 8888) grades each question via a pluggable `MarkScheme` (LLM-judge for the POC). Pure-Python; both models are OpenAI-compatible HTTP endpoints already serving on the DGX.

**Tech Stack:** Python 3.12 (via `uv`), `pydantic` v2, `httpx`, `pytest`; `pdftoppm` (poppler, already installed) for PDF→PNG. No PyMuPDF/OpenCV — a plain render is sufficient (verified).

## Global Constraints

- Python **3.12** managed by `uv` (local default is 3.14 and lacks wheels — do not use it).
- Dependencies limited to: `pydantic>=2`, `httpx`, `pytest`. Rendering via `pdftoppm` subprocess. No other runtime deps.
- DGX endpoints (verified live 2026-06-22), defaults in `config.py`:
  - Transcriber: `http://192.168.10.246:8003/v1`, model `qwen3-vl`
  - Grader: `http://192.168.10.246:8888/v1`, model `qwen3.6-35b`
- Student PDFs contain minors' names → never commit them or rendered pages (`.gitignore` already covers `*.pdf`, `*.png`, `out/`).
- All cross-stage data passes as the Pydantic models in `examgrader/schemas.py`. No dicts across module boundaries.
- TDD: write the failing test first, watch it fail, implement minimal code, watch it pass, commit. LLM HTTP calls are always mocked in unit tests — never hit the DGX from `pytest`.

---

## File Structure

```
pyproject.toml                 # uv project + deps + pytest config
examgrader/
  __init__.py
  config.py                    # Settings dataclass + SETTINGS singleton
  schemas.py                   # Pydantic models (the stage interfaces)
  llm_client.py                # OpenAI-compatible HTTP client w/ retry + JSON extraction
  pdf_to_images.py             # pdftoppm render → page PNG paths; blank-page skip
  transcriber.py               # page images → TranscribedPaper (calls VLM)
  grader.py                    # MarkScheme interface + LLMJudge → GradedPaper
  report.py                    # GradedPaper → results.json + report.md
  cli.py                       # orchestrate full pipeline; persist transcripts
tests/
  conftest.py                  # shared fixtures (golden transcript, fake clients)
  fixtures/golden_transcript.json
  test_schemas.py
  test_llm_client.py
  test_pdf_to_images.py
  test_transcriber.py
  test_grader.py
  test_report.py
grade.py                       # thin entrypoint -> examgrader.cli.main()
README.md
```

---

## Task 1: Project scaffold + config

**Files:**
- Create: `pyproject.toml`, `examgrader/__init__.py`, `examgrader/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `examgrader.config.Settings` (frozen dataclass) and `SETTINGS` singleton with fields `vlm_base_url:str`, `vlm_model:str`, `grader_base_url:str`, `grader_model:str`, `render_dpi:int`, `request_timeout:float`, `max_retries:int`, `out_dir:str`.

- [ ] **Step 1: Create the uv project and venv**

```bash
cd /Users/pascal/Development/wael-exames
uv python install 3.12
uv init --no-workspace --name examgrader --python 3.12
uv add "pydantic>=2" httpx
uv add --dev pytest
```

- [ ] **Step 2: Replace `pyproject.toml` pytest config block**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:

```python
from examgrader.config import SETTINGS

def test_settings_defaults_point_at_dgx():
    assert SETTINGS.vlm_base_url == "http://192.168.10.246:8003/v1"
    assert SETTINGS.vlm_model == "qwen3-vl"
    assert SETTINGS.grader_base_url == "http://192.168.10.246:8888/v1"
    assert SETTINGS.grader_model == "qwen3.6-35b"
    assert SETTINGS.render_dpi == 200
```

- [ ] **Step 4: Run test, verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.config'`

- [ ] **Step 5: Implement `examgrader/__init__.py` (empty) and `examgrader/config.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vlm_base_url: str = "http://192.168.10.246:8003/v1"
    vlm_model: str = "qwen3-vl"
    grader_base_url: str = "http://192.168.10.246:8888/v1"
    grader_model: str = "qwen3.6-35b"
    render_dpi: int = 200
    request_timeout: float = 180.0
    max_retries: int = 3
    out_dir: str = "out"


SETTINGS = Settings()
```

- [ ] **Step 6: Run test, verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock examgrader/__init__.py examgrader/config.py tests/test_config.py
git commit -m "feat: project scaffold + DGX config"
```

---

## Task 2: Schemas (stage interfaces)

**Files:**
- Create: `examgrader/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `TranscribedQuestion(section:str|None, question_no:str, max_marks:float, question_text:str, student_answer:str, read_confidence:float[0..1])`
  - `TranscribedPaper(subject:str, source_pdf:str, questions:list[TranscribedQuestion])`
  - `GradedQuestion(question_no:str, section:str|None, max_marks:float, awarded_marks:float, student_answer:str, justification:str, grade_confidence:float[0..1], flags:list[str])`
  - `GradedPaper(subject:str, source_pdf:str, questions:list[GradedQuestion], section_totals:dict[str,float], total:float, max_total:float)`

- [ ] **Step 1: Write the failing test**

`tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from examgrader.schemas import (
    TranscribedQuestion, TranscribedPaper, GradedQuestion, GradedPaper,
)


def test_transcribed_question_roundtrip():
    q = TranscribedQuestion(
        section="A", question_no="1a", max_marks=5,
        question_text="(-3) x (-2) = -6", student_answer="False", read_confidence=0.9,
    )
    assert q.max_marks == 5.0
    assert q.read_confidence == 0.9


def test_read_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        TranscribedQuestion(
            question_no="1", max_marks=1, question_text="x",
            student_answer="y", read_confidence=1.5,
        )


def test_graded_paper_defaults_flags_empty():
    g = GradedQuestion(
        question_no="1", max_marks=5, awarded_marks=3,
        student_answer="False", justification="ok", grade_confidence=0.8,
    )
    assert g.flags == []


def test_papers_nest():
    tp = TranscribedPaper(
        subject="Math", source_pdf="Math paper.pdf",
        questions=[TranscribedQuestion(question_no="1", max_marks=1,
                   question_text="x", student_answer="y", read_confidence=0.5)],
    )
    assert tp.questions[0].question_no == "1"
    gp = GradedPaper(
        subject="Math", source_pdf="Math paper.pdf", questions=[],
        section_totals={"A": 3.0}, total=3.0, max_total=100.0,
    )
    assert gp.section_totals["A"] == 3.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.schemas'`

- [ ] **Step 3: Implement `examgrader/schemas.py`**

```python
from pydantic import BaseModel, Field


class TranscribedQuestion(BaseModel):
    section: str | None = None
    question_no: str
    max_marks: float
    question_text: str
    student_answer: str
    read_confidence: float = Field(ge=0.0, le=1.0)


class TranscribedPaper(BaseModel):
    subject: str
    source_pdf: str
    questions: list[TranscribedQuestion]


class GradedQuestion(BaseModel):
    question_no: str
    section: str | None = None
    max_marks: float
    awarded_marks: float
    student_answer: str
    justification: str
    grade_confidence: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)


class GradedPaper(BaseModel):
    subject: str
    source_pdf: str
    questions: list[GradedQuestion]
    section_totals: dict[str, float]
    total: float
    max_total: float
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add examgrader/schemas.py tests/test_schemas.py
git commit -m "feat: pydantic stage interfaces"
```

---

## Task 3: LLM client (HTTP + retry + JSON extraction)

**Files:**
- Create: `examgrader/llm_client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `extract_json(text:str) -> dict | list` — pull the first JSON value out of a model reply (handles ```` ```json ```` fences and leading prose).
  - `LLMClient(base_url:str, model:str, timeout:float=180.0, max_retries:int=3)` with method `chat_json(content, *, max_tokens:int=1500, temperature:float=0.0) -> dict|list`. `content` is the OpenAI message `content` (str or list of content parts). It POSTs to `{base_url}/chat/completions`, reads `choices[0].message.content`, returns `extract_json(...)`. Retries on any exception up to `max_retries`, then raises `RuntimeError`.
  - `image_part(png_path:str) -> dict` — returns `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`.
  - `text_part(text:str) -> dict` — returns `{"type":"text","text":text}`.

- [ ] **Step 1: Write the failing test**

`tests/test_llm_client.py`:

```python
import json
import pytest
from examgrader import llm_client
from examgrader.llm_client import LLMClient, extract_json, text_part


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence_and_prose():
    raw = 'Here you go:\n```json\n[{"x": 2}]\n```\nthanks'
    assert extract_json(raw) == [{"x": 2}]


def test_text_part_shape():
    assert text_part("hi") == {"type": "text", "text": "hi"}


def test_chat_json_parses_reply(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr(llm_client.httpx, "post", fake_post)
    c = LLMClient("http://x/v1", "m")
    out = c.chat_json("hello", max_tokens=10)
    assert out == {"ok": True}
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["payload"]["model"] == "m"
    assert captured["payload"]["messages"][0]["content"] == "hello"


def test_chat_json_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_client.httpx, "post", fake_post)
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)
    c = LLMClient("http://x/v1", "m", max_retries=3)
    with pytest.raises(RuntimeError):
        c.chat_json("hi")
    assert calls["n"] == 3
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.llm_client'`

- [ ] **Step 3: Implement `examgrader/llm_client.py`**

```python
import base64
import json
import time

import httpx


def extract_json(text: str):
    """Return the first JSON object/array found in a model reply."""
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence (``` or ```json) and closing fence
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if "```" in text:
            text = text[: text.index("```")]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back: scan for the first balanced { } or [ ]
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in reply: {text[:200]!r}")
    start = min(starts)
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in reply: {text[:200]!r}")


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(png_path: str) -> dict:
    data = base64.b64encode(open(png_path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def chat_json(self, content, *, max_tokens: int = 1500, temperature: float = 0.0):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = httpx.post(
                    f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
                )
                r.raise_for_status()
                reply = r.json()["choices"][0]["message"]["content"]
                return extract_json(reply)
            except Exception as e:  # noqa: BLE001 - retry on any failure
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"LLM call to {self.base_url} failed after {self.max_retries} attempts: {last_err}"
        )
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add examgrader/llm_client.py tests/test_llm_client.py
git commit -m "feat: OpenAI-compatible LLM client with retry + JSON extraction"
```

---

## Task 4: PDF → page images

**Files:**
- Create: `examgrader/pdf_to_images.py`
- Test: `tests/test_pdf_to_images.py`

**Interfaces:**
- Consumes: `SETTINGS.render_dpi`.
- Produces:
  - `render_pdf(pdf_path:str, out_dir:str, dpi:int|None=None) -> list[str]` — runs `pdftoppm -png -r <dpi> <pdf> <out_dir>/page`, returns sorted list of generated PNG paths.
  - `is_blank(png_path:str, threshold:float=0.997) -> bool` — True if the page is ~empty (fraction of near-white pixels ≥ threshold), using only stdlib by reading the PNG via `pdftoppm`-produced grayscale. Implemented by shelling to `pdftoppm`'s sibling tool is overkill; instead read PNG bytes with a tiny pure-python check: re-render not needed — use the file size heuristic AND a pixel check via `pnmto`/no. **Use the approach in Step 3 (ImageMagick `identify` mean).**
  - `content_pages(pdf_path:str, out_dir:str, dpi:int|None=None) -> list[str]` — render then drop blanks.

- [ ] **Step 1: Write the failing test**

`tests/test_pdf_to_images.py`:

```python
import os
import pytest
from examgrader import pdf_to_images

MATH = "Math paper.pdf"
pytestmark = pytest.mark.skipif(not os.path.exists(MATH), reason="sample PDF absent")


def test_render_pdf_produces_pngs(tmp_path):
    pages = pdf_to_images.render_pdf(MATH, str(tmp_path), dpi=120)
    assert len(pages) >= 3
    assert all(p.endswith(".png") for p in pages)
    assert pages == sorted(pages)
    assert all(os.path.getsize(p) > 0 for p in pages)


def test_content_pages_drops_blanks(tmp_path):
    all_pages = pdf_to_images.render_pdf(MATH, str(tmp_path), dpi=120)
    content = pdf_to_images.content_pages(MATH, str(tmp_path), dpi=120)
    assert 0 < len(content) <= len(all_pages)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_pdf_to_images.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.pdf_to_images'`

- [ ] **Step 3: Implement `examgrader/pdf_to_images.py`**

```python
import glob
import os
import subprocess

from examgrader.config import SETTINGS


def render_pdf(pdf_path: str, out_dir: str, dpi: int | None = None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    dpi = dpi or SETTINGS.render_dpi
    prefix = os.path.join(out_dir, "page")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), pdf_path, prefix],
        check=True, capture_output=True,
    )
    return sorted(glob.glob(prefix + "-*.png"))


def is_blank(png_path: str, threshold: float = 0.985) -> bool:
    """Blank if the page's mean brightness is near white (scanned empty page)."""
    out = subprocess.run(
        ["magick", "identify", "-format", "%[fx:mean]", png_path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    mean = float(out)  # 0..1, 1.0 == pure white
    return mean >= threshold


def content_pages(pdf_path: str, out_dir: str, dpi: int | None = None) -> list[str]:
    return [p for p in render_pdf(pdf_path, out_dir, dpi) if not is_blank(p)]
```

*Note for implementer:* `magick` (ImageMagick) and `pdftoppm` (poppler) are installed system tools. The blank `threshold` of 0.985 was chosen because NESA scans have light speckle (not pure 1.0). If a known-content page is wrongly dropped during the Task 9 integration run, raise the threshold toward 0.995.

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_pdf_to_images.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add examgrader/pdf_to_images.py tests/test_pdf_to_images.py
git commit -m "feat: pdftoppm rendering + blank-page detection"
```

---

## Task 5: Transcriber (VLM → TranscribedPaper)

**Files:**
- Create: `examgrader/transcriber.py`, `tests/conftest.py`, `tests/fixtures/golden_transcript.json`
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: `LLMClient`, `text_part`, `image_part`, `TranscribedQuestion`, `TranscribedPaper`, `SETTINGS`.
- Produces:
  - `TRANSCRIBE_PROMPT: str`
  - `transcribe_page(client:LLMClient, png_path:str) -> list[dict]` — sends one page image, returns the raw list of question dicts the VLM produced (keys: `section, question_no, max_marks, question_text, student_answer, read_confidence`).
  - `transcribe_paper(client:LLMClient, png_paths:list[str], subject:str, source_pdf:str) -> TranscribedPaper` — calls `transcribe_page` per page, flattens, validates each into `TranscribedQuestion`, returns `TranscribedPaper`. A page whose call/parse fails is skipped (logged to stderr), not fatal.

- [ ] **Step 1: Create the golden fixture `tests/fixtures/golden_transcript.json`**

```json
{
  "subject": "Math",
  "source_pdf": "Math paper.pdf",
  "questions": [
    {"section": "1", "question_no": "1a", "max_marks": 1, "question_text": "(-3) x (-2) = -6", "student_answer": "False", "read_confidence": 0.95},
    {"section": "1", "question_no": "1b", "max_marks": 1, "question_text": "(-2) + (-3) = -5", "student_answer": "True", "read_confidence": 0.95},
    {"section": "2", "question_no": "2a", "max_marks": 1, "question_text": "The money borrowed, saved or lent is ___", "student_answer": "Principal", "read_confidence": 0.9},
    {"section": "3", "question_no": "3a", "max_marks": 2, "question_text": "The distance around the circle is ___", "student_answer": "Circumference", "read_confidence": 0.9}
  ]
}
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import json
import pathlib
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_transcript_dict():
    return json.loads((FIXTURES / "golden_transcript.json").read_text())


class FakeClient:
    """Stand-in for LLMClient: returns queued JSON values, records calls."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat_json(self, content, *, max_tokens=1500, temperature=0.0):
        self.calls.append(content)
        return self._replies.pop(0)


@pytest.fixture
def fake_client_factory():
    return FakeClient
```

- [ ] **Step 3: Write the failing test**

`tests/test_transcriber.py`:

```python
from examgrader import transcriber
from examgrader.schemas import TranscribedPaper


def test_transcribe_paper_builds_model(fake_client_factory, golden_transcript_dict, tmp_path):
    page_png = tmp_path / "page-01.png"
    page_png.write_bytes(b"\x89PNG\r\n")  # bytes only need to be readable by image_part
    client = fake_client_factory([golden_transcript_dict["questions"]])
    tp = transcriber.transcribe_paper(client, [str(page_png)], "Math", "Math paper.pdf")
    assert isinstance(tp, TranscribedPaper)
    assert len(tp.questions) == 4
    assert tp.questions[0].student_answer == "False"
    assert tp.questions[2].student_answer == "Principal"


def test_transcribe_paper_skips_failing_page(fake_client_factory, golden_transcript_dict, tmp_path):
    p1 = tmp_path / "page-01.png"; p1.write_bytes(b"\x89PNG\r\n")
    p2 = tmp_path / "page-02.png"; p2.write_bytes(b"\x89PNG\r\n")

    class Flaky:
        def __init__(self): self.n = 0
        def chat_json(self, content, **kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("vlm down")
            return golden_transcript_dict["questions"]

    tp = transcriber.transcribe_paper(Flaky(), [str(p1), str(p2)], "Math", "Math paper.pdf")
    assert len(tp.questions) == 4  # page 1 skipped, page 2 parsed
```

- [ ] **Step 4: Run test, verify it fails**

Run: `uv run pytest tests/test_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.transcriber'`

- [ ] **Step 5: Implement `examgrader/transcriber.py`**

```python
import sys

from examgrader.llm_client import image_part, text_part
from examgrader.schemas import TranscribedPaper, TranscribedQuestion

TRANSCRIBE_PROMPT = (
    "This is one scanned page of a primary-school exam with PRINTED questions and "
    "HANDWRITTEN student answers. Extract every question that has an answer space on "
    "this page. Return ONLY a JSON array; each element has keys: "
    '"section" (the section letter/number or null), '
    '"question_no" (e.g. "1a"), '
    '"max_marks" (number from the printed "(N marks)" label, or 0 if none shown), '
    '"question_text" (the printed question, concise), '
    '"student_answer" (the handwriting transcribed exactly; empty string if blank), '
    '"read_confidence" (0..1, your confidence in reading the handwriting). '
    "Do not invent questions that are not on this page."
)


def transcribe_page(client, png_path: str) -> list[dict]:
    content = [text_part(TRANSCRIBE_PROMPT), image_part(png_path)]
    result = client.chat_json(content, max_tokens=2000)
    return result if isinstance(result, list) else result.get("questions", [])


def transcribe_paper(client, png_paths, subject: str, source_pdf: str) -> TranscribedPaper:
    questions: list[TranscribedQuestion] = []
    for path in png_paths:
        try:
            for raw in transcribe_page(client, path):
                questions.append(TranscribedQuestion(**raw))
        except Exception as e:  # noqa: BLE001 - one bad page must not sink the paper
            print(f"[transcriber] skipped {path}: {e}", file=sys.stderr)
    return TranscribedPaper(subject=subject, source_pdf=source_pdf, questions=questions)
```

- [ ] **Step 6: Run test, verify it passes**

Run: `uv run pytest tests/test_transcriber.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add examgrader/transcriber.py tests/conftest.py tests/fixtures/golden_transcript.json tests/test_transcriber.py
git commit -m "feat: VLM transcriber with per-page isolation"
```

---

## Task 6: Grader (MarkScheme + LLMJudge → GradedPaper)

**Files:**
- Create: `examgrader/grader.py`
- Test: `tests/test_grader.py`

**Interfaces:**
- Consumes: `LLMClient`, `text_part`, `TranscribedPaper`, `TranscribedQuestion`, `GradedQuestion`, `GradedPaper`.
- Produces:
  - `MarkScheme` (Protocol): `grade_question(q:TranscribedQuestion) -> GradedQuestion`.
  - `LLMJudge(client:LLMClient)` implementing `MarkScheme`. Builds a judging prompt, calls `client.chat_json`, maps reply `{awarded_marks, justification, grade_confidence}` to a `GradedQuestion`. Clamps `awarded_marks` to `[0, max_marks]`. On failure, returns a `GradedQuestion` with `awarded_marks=0`, `grade_confidence=0.0`, `flags=["grading_failed"]`. Adds `flags=["low_read_confidence"]` when `q.read_confidence < 0.5`.
  - `JUDGE_PROMPT: str`.
  - `grade_paper(scheme:MarkScheme, paper:TranscribedPaper, max_total:float=100.0) -> GradedPaper` — grades every question, computes `section_totals` (sum of awarded by `section`, key `"?"` when section is None) and `total`.

- [ ] **Step 1: Write the failing test**

`tests/test_grader.py`:

```python
from examgrader import grader
from examgrader.schemas import TranscribedPaper, TranscribedQuestion, GradedQuestion


def _q(no, max_marks, ans, conf=0.9, section="A", text="q"):
    return TranscribedQuestion(section=section, question_no=no, max_marks=max_marks,
                               question_text=text, student_answer=ans, read_confidence=conf)


def test_llmjudge_maps_and_clamps(fake_client_factory):
    client = fake_client_factory([
        {"awarded_marks": 99, "justification": "correct", "grade_confidence": 0.9},
    ])
    judge = grader.LLMJudge(client)
    g = judge.grade_question(_q("1a", 5, "False"))
    assert isinstance(g, GradedQuestion)
    assert g.awarded_marks == 5  # clamped to max_marks
    assert g.justification == "correct"


def test_llmjudge_flags_low_read_confidence(fake_client_factory):
    client = fake_client_factory([
        {"awarded_marks": 1, "justification": "ok", "grade_confidence": 0.8},
    ])
    g = grader.LLMJudge(client).grade_question(_q("2a", 1, "Principal", conf=0.3))
    assert "low_read_confidence" in g.flags


def test_llmjudge_handles_call_failure():
    class Boom:
        def chat_json(self, *a, **k): raise RuntimeError("down")
    g = grader.LLMJudge(Boom()).grade_question(_q("3a", 2, "Circumference"))
    assert g.awarded_marks == 0
    assert "grading_failed" in g.flags


def test_grade_paper_totals(fake_client_factory):
    client = fake_client_factory([
        {"awarded_marks": 1, "justification": "a", "grade_confidence": 1.0},
        {"awarded_marks": 0, "justification": "b", "grade_confidence": 1.0},
        {"awarded_marks": 2, "justification": "c", "grade_confidence": 1.0},
    ])
    paper = TranscribedPaper(subject="Math", source_pdf="Math paper.pdf", questions=[
        _q("1a", 1, "False", section="A"),
        _q("1b", 1, "True", section="A"),
        _q("3a", 2, "Circumference", section="B"),
    ])
    gp = grader.grade_paper(grader.LLMJudge(client), paper)
    assert gp.total == 3.0
    assert gp.section_totals == {"A": 1.0, "B": 2.0}
    assert gp.max_total == 100.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_grader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.grader'`

- [ ] **Step 3: Implement `examgrader/grader.py`**

```python
from typing import Protocol

from examgrader.llm_client import text_part
from examgrader.schemas import (
    GradedPaper, GradedQuestion, TranscribedPaper, TranscribedQuestion,
)

JUDGE_PROMPT = (
    "You are grading one exam question. You are given the question, its maximum marks, "
    "and the student's answer (already transcribed from handwriting). Decide the correct "
    "answer yourself, then award marks. For questions that show working, give partial "
    "credit for correct method. Return ONLY a JSON object with keys: "
    '"awarded_marks" (number, 0..max), '
    '"justification" (one sentence), '
    '"grade_confidence" (0..1).'
)


class MarkScheme(Protocol):
    def grade_question(self, q: TranscribedQuestion) -> GradedQuestion: ...


class LLMJudge:
    def __init__(self, client):
        self.client = client

    def grade_question(self, q: TranscribedQuestion) -> GradedQuestion:
        flags: list[str] = []
        if q.read_confidence < 0.5:
            flags.append("low_read_confidence")
        prompt = (
            f"{JUDGE_PROMPT}\n\n"
            f"Question {q.question_no}: {q.question_text}\n"
            f"Maximum marks: {q.max_marks}\n"
            f"Student answer: {q.student_answer!r}"
        )
        try:
            r = self.client.chat_json([text_part(prompt)], max_tokens=400)
            awarded = max(0.0, min(float(r["awarded_marks"]), float(q.max_marks)))
            return GradedQuestion(
                question_no=q.question_no, section=q.section, max_marks=q.max_marks,
                awarded_marks=awarded, student_answer=q.student_answer,
                justification=str(r.get("justification", "")),
                grade_confidence=float(r.get("grade_confidence", 0.0)),
                flags=flags,
            )
        except Exception as e:  # noqa: BLE001 - isolate one question's failure
            return GradedQuestion(
                question_no=q.question_no, section=q.section, max_marks=q.max_marks,
                awarded_marks=0.0, student_answer=q.student_answer,
                justification=f"grading failed: {e}", grade_confidence=0.0,
                flags=flags + ["grading_failed"],
            )


def grade_paper(scheme: MarkScheme, paper: TranscribedPaper, max_total: float = 100.0) -> GradedPaper:
    graded = [scheme.grade_question(q) for q in paper.questions]
    section_totals: dict[str, float] = {}
    for g in graded:
        key = g.section or "?"
        section_totals[key] = section_totals.get(key, 0.0) + g.awarded_marks
    total = sum(g.awarded_marks for g in graded)
    return GradedPaper(
        subject=paper.subject, source_pdf=paper.source_pdf, questions=graded,
        section_totals=section_totals, total=total, max_total=max_total,
    )
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_grader.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add examgrader/grader.py tests/test_grader.py
git commit -m "feat: grader with pluggable MarkScheme (LLM-judge)"
```

---

## Task 7: Report (JSON + Markdown)

**Files:**
- Create: `examgrader/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `GradedPaper`.
- Produces:
  - `to_json(paper:GradedPaper) -> str` — pretty JSON of the paper.
  - `to_markdown(paper:GradedPaper) -> str` — readable report: header with subject + total/max, a table of `question_no | section | awarded/max | confidence | flags | justification`, then a section-totals summary. Questions with any `flags` are marked with a ⚠ in the table.
  - `write_report(paper:GradedPaper, out_dir:str) -> tuple[str,str]` — writes `<out_dir>/<stem>.results.json` and `<out_dir>/<stem>.report.md` (stem from `source_pdf`), returns the two paths.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:

```python
import json
from examgrader import report
from examgrader.schemas import GradedPaper, GradedQuestion


def _paper():
    return GradedPaper(
        subject="Math", source_pdf="Math paper.pdf",
        questions=[
            GradedQuestion(question_no="1a", section="A", max_marks=5, awarded_marks=5,
                           student_answer="False", justification="correct", grade_confidence=0.9),
            GradedQuestion(question_no="2a", section="B", max_marks=1, awarded_marks=0,
                           student_answer="x", justification="wrong", grade_confidence=0.4,
                           flags=["low_read_confidence"]),
        ],
        section_totals={"A": 5.0, "B": 0.0}, total=5.0, max_total=100.0,
    )


def test_to_json_roundtrips():
    data = json.loads(report.to_json(_paper()))
    assert data["total"] == 5.0
    assert data["questions"][0]["question_no"] == "1a"


def test_to_markdown_has_header_and_flag():
    md = report.to_markdown(_paper())
    assert "Math" in md
    assert "5" in md and "100" in md
    assert "1a" in md and "2a" in md
    assert "⚠" in md  # flagged question marked


def test_write_report_creates_files(tmp_path):
    j, m = report.write_report(_paper(), str(tmp_path))
    assert j.endswith("Math paper.results.json")
    assert m.endswith("Math paper.report.md")
    assert json.loads(open(j).read())["subject"] == "Math"
    assert "Math" in open(m).read()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.report'`

- [ ] **Step 3: Implement `examgrader/report.py`**

```python
import os

from examgrader.schemas import GradedPaper


def to_json(paper: GradedPaper) -> str:
    return paper.model_dump_json(indent=2)


def to_markdown(paper: GradedPaper) -> str:
    lines = [
        f"# {paper.subject} — graded ({paper.source_pdf})",
        "",
        f"**Total: {paper.total:g} / {paper.max_total:g}**",
        "",
        "| Q | Section | Marks | Conf | Flags | Justification |",
        "|---|---------|-------|------|-------|---------------|",
    ]
    for q in paper.questions:
        warn = "⚠ " if q.flags else ""
        flags = ", ".join(q.flags) if q.flags else ""
        just = q.justification.replace("|", "\\|")
        lines.append(
            f"| {warn}{q.question_no} | {q.section or ''} | "
            f"{q.awarded_marks:g}/{q.max_marks:g} | {q.grade_confidence:g} | "
            f"{flags} | {just} |"
        )
    lines += ["", "## Section totals", ""]
    for sec, tot in sorted(paper.section_totals.items()):
        lines.append(f"- {sec}: {tot:g}")
    return "\n".join(lines) + "\n"


def write_report(paper: GradedPaper, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(paper.source_pdf))[0]
    json_path = os.path.join(out_dir, f"{stem}.results.json")
    md_path = os.path.join(out_dir, f"{stem}.report.md")
    with open(json_path, "w") as f:
        f.write(to_json(paper))
    with open(md_path, "w") as f:
        f.write(to_markdown(paper))
    return json_path, md_path
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add examgrader/report.py tests/test_report.py
git commit -m "feat: JSON + Markdown report generation"
```

---

## Task 8: CLI orchestration

**Files:**
- Create: `examgrader/cli.py`, `grade.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above + `SETTINGS`.
- Produces:
  - `grade_pdf(pdf_path:str, subject:str, *, out_dir:str|None=None, vlm_client=None, grader_client=None) -> GradedPaper` — full pipeline: `content_pages` → `transcribe_paper` (persist `<stem>.transcript.json` to out_dir) → `grade_paper(LLMJudge)` → `write_report`. `vlm_client`/`grader_client` default to real `LLMClient`s built from `SETTINGS`; tests inject fakes.
  - `main(argv:list[str]|None=None) -> int` — argparse: `grade.py <pdf> --subject NAME [--out DIR]`. Subject defaults to the PDF stem. Prints the report paths + total.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
import json
import os
from examgrader import cli
from examgrader.schemas import GradedPaper


def test_grade_pdf_pipeline(monkeypatch, fake_client_factory, golden_transcript_dict, tmp_path):
    # stub rendering so we don't need poppler in this unit test
    fake_pages = [str(tmp_path / "page-01.png")]
    open(fake_pages[0], "wb").write(b"\x89PNG\r\n")
    monkeypatch.setattr(cli, "content_pages", lambda *a, **k: fake_pages)

    vlm = fake_client_factory([golden_transcript_dict["questions"]])
    # one grader reply per question (4 in the golden fixture)
    grader_replies = [{"awarded_marks": 1, "justification": "ok", "grade_confidence": 1.0}] * 4
    grd = fake_client_factory(grader_replies)

    gp = cli.grade_pdf("Math paper.pdf", "Math", out_dir=str(tmp_path),
                       vlm_client=vlm, grader_client=grd)
    assert isinstance(gp, GradedPaper)
    assert gp.total == 4.0
    # transcript + report artifacts written
    assert os.path.exists(tmp_path / "Math paper.transcript.json")
    assert os.path.exists(tmp_path / "Math paper.results.json")
    assert json.loads((tmp_path / "Math paper.transcript.json").read_text())["questions"]
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examgrader.cli'`

- [ ] **Step 3: Implement `examgrader/cli.py` and `grade.py`**

`examgrader/cli.py`:

```python
import argparse
import os
import sys

from examgrader.config import SETTINGS
from examgrader.grader import LLMJudge, grade_paper
from examgrader.llm_client import LLMClient
from examgrader.pdf_to_images import content_pages
from examgrader.report import write_report
from examgrader.schemas import GradedPaper
from examgrader.transcriber import transcribe_paper


def grade_pdf(pdf_path, subject, *, out_dir=None, vlm_client=None, grader_client=None) -> GradedPaper:
    out_dir = out_dir or SETTINGS.out_dir
    os.makedirs(out_dir, exist_ok=True)
    vlm_client = vlm_client or LLMClient(
        SETTINGS.vlm_base_url, SETTINGS.vlm_model, SETTINGS.request_timeout, SETTINGS.max_retries
    )
    grader_client = grader_client or LLMClient(
        SETTINGS.grader_base_url, SETTINGS.grader_model, SETTINGS.request_timeout, SETTINGS.max_retries
    )
    stem = os.path.splitext(os.path.basename(pdf_path))[0]

    pages = content_pages(pdf_path, os.path.join(out_dir, f"{stem}_pages"))
    transcript = transcribe_paper(vlm_client, pages, subject, os.path.basename(pdf_path))
    with open(os.path.join(out_dir, f"{stem}.transcript.json"), "w") as f:
        f.write(transcript.model_dump_json(indent=2))

    paper = grade_paper(LLMJudge(grader_client), transcript)
    write_report(paper, out_dir)
    return paper


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade a scanned exam PDF on the DGX.")
    ap.add_argument("pdf")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--out", default=SETTINGS.out_dir)
    args = ap.parse_args(argv)
    subject = args.subject or os.path.splitext(os.path.basename(args.pdf))[0]
    paper = grade_pdf(args.pdf, subject, out_dir=args.out)
    print(f"{subject}: {paper.total:g}/{paper.max_total:g}")
    print(f"Reports written under {args.out}/", file=sys.stderr)
    return 0
```

`grade.py`:

```python
import sys
from examgrader.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add examgrader/cli.py grade.py tests/test_cli.py
git commit -m "feat: CLI pipeline orchestration with transcript persistence"
```

---

## Task 9: Live integration run on the two real PDFs

**Files:**
- Create: `README.md`
- No new tests (this task hits the live DGX; it is a manual verification, not a `pytest`).

**Interfaces:**
- Consumes: the full pipeline + live DGX endpoints.

- [ ] **Step 1: Confirm the DGX endpoints are reachable**

```bash
curl -s -o /dev/null -w "vlm:%{http_code}\n" http://192.168.10.246:8003/v1/models
curl -s -o /dev/null -w "grader:%{http_code}\n" http://192.168.10.246:8888/v1/models
```
Expected: `vlm:200` and `grader:200`. If not 200, the models need restarting on the DGX (see the design spec's Environment section / `~/launch-qwen3-vl.sh`).

- [ ] **Step 2: Grade the Math paper**

```bash
uv run python grade.py "Math paper.pdf" --subject Math --out out
```
Expected: prints `Math: <score>/100`; `out/Math paper.transcript.json`, `out/Math paper.results.json`, `out/Math paper.report.md` exist.

- [ ] **Step 3: Spot-check the Math transcript against the scan**

Open `out/Math paper.transcript.json`. Verify Q1 reads `False, True, True, False, True`; Q2 `Principal, Rate, Time`; Q3a `Circumference` (these are confirmed-correct ground truth from the design verification). If the transcript is wrong, the issue is the VLM/render — raise `--out` DPI by editing `SETTINGS.render_dpi` to 250 and re-run. Note any mismatch in the README.

- [ ] **Step 4: Grade the English paper**

```bash
uv run python grade.py "English paper.pdf" --subject English --out out
```
Expected: prints `English: <score>/100`; three artifacts written. The English paper has open-ended composition — expect the grader to award partial marks with `low_read_confidence`/justification on subjective items.

- [ ] **Step 5: Review both Markdown reports**

```bash
cat "out/Math paper.report.md"
cat "out/English paper.report.md"
```
Confirm: per-question marks ≤ max, section totals sum to the printed allocations (Math sections, English A=20/B=25/C=40/D=15), flagged questions where handwriting was uncertain.

- [ ] **Step 6: Write `README.md`**

```markdown
# wael-exames — local exam grading

Grades scanned NESA exam PDFs on the DGX Spark using a two-stage local-LLM pipeline:
`pdftoppm` render → `qwen3-vl` (handwriting transcription, port 8003) →
`qwen3.6-35b` (grading, port 8888) → JSON + Markdown report.

## Run

    uv run python grade.py "Math paper.pdf" --subject Math --out out

Outputs `out/<stem>.transcript.json`, `out/<stem>.results.json`, `out/<stem>.report.md`.

## Test

    uv run pytest

Unit tests mock the LLM calls; no DGX needed. The live run (Task 9 in the plan) hits the DGX.

## Notes
- Endpoints + DPI live in `examgrader/config.py`.
- POC uses LLM-as-judge; production swaps `LLMJudge` for a marking-guide `MarkScheme`.
- Student PDFs and rendered pages are gitignored (contain minors' names).
```

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: README + live integration verified on both papers"
```

---

## Self-Review notes (already applied)

- **Spec coverage:** pdf_to_images (T4), transcriber/VLM (T5), grader + MarkScheme interface for production swap (T6), report JSON+MD (T7), error handling/retry/per-item isolation (T3/T5/T6), confidence flags (T6), persisted transcripts (T8), 2-PDF end-to-end (T9). LLM-judge-now / marking-guide-later is the `MarkScheme` Protocol.
- **Deviations from spec (intentional, lower-risk):** PyMuPDF+OpenCV replaced by `pdftoppm` (no preprocessing) — verified sufficient; Python pinned to 3.12 via `uv` (local 3.14 lacks wheels); vision model is `qwen3-vl` on port 8003 (verified) rather than the originally-guessed Qwen2.5-VL/8889.
- **Type consistency:** `TranscribedQuestion`/`GradedQuestion` field names are used identically across transcriber, grader, report, cli. `chat_json` signature identical in `LLMClient` and `FakeClient`.
