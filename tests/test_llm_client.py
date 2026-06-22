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


def test_image_part_builds_data_url(tmp_path):
    from examgrader.llm_client import image_part
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nABC")
    part = image_part(str(p))
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    # round-trips to the original bytes
    import base64
    b64 = part["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(b64) == b"\x89PNG\r\n\x1a\nABC"


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
