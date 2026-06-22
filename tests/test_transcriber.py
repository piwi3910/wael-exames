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
