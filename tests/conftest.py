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
