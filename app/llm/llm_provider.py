import logging
from collections.abc import Iterator, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings


logger = logging.getLogger(__name__)


class LLMProvider:
    def __init__(self, settings: Settings) -> None:
        self._client = ChatOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_chat_model,
            temperature=settings.openai_temperature,
            timeout=settings.openai_request_timeout,
            streaming=True,
        )
        self._max_retries = settings.llm_max_retries
        self._retry_backoff_seconds = settings.llm_retry_backoff_seconds

    def _build_retrying(self) -> Retrying:
        return Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._retry_backoff_seconds, min=self._retry_backoff_seconds),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

    def generate(self, messages: Sequence[BaseMessage]) -> str:
        try:
            for attempt in self._build_retrying():
                with attempt:
                    response = self._client.invoke(list(messages))
        except RetryError as exc:
            logger.exception("LLM invocation exhausted all retries.")
            raise exc.last_attempt.exception() from exc

        if isinstance(response.content, str):
            return response.content.strip()

        return "".join(part.get("text", "") for part in response.content).strip()

    def stream_generate(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        try:
            for attempt in self._build_retrying():
                with attempt:
                    stream = self._client.stream(list(messages))
                    break
        except RetryError as exc:
            logger.exception("LLM stream initialization exhausted all retries.")
            raise exc.last_attempt.exception() from exc

        for chunk in stream:
            content = chunk.content
            if isinstance(content, str) and content:
                yield content
                continue

            if isinstance(content, list):
                for part in content:
                    text = part.get("text", "")
                    if text:
                        yield text