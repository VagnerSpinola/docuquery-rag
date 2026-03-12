from langchain_core.messages import HumanMessage

from app.llm.llm_provider import LLMProvider


class StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return StubResponse("Recovered answer")


def test_llm_provider_retries_transient_failures() -> None:
    provider = object.__new__(LLMProvider)
    provider._client = FlakyClient()
    provider._max_retries = 3
    provider._retry_backoff_seconds = 0

    answer = provider.generate([HumanMessage(content="hello")])

    assert answer == "Recovered answer"
    assert provider._client.calls == 2