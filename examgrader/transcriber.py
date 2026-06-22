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
