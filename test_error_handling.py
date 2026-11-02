"""Model API error handling tests — no API key or network needed."""

import unittest
from unittest.mock import Mock

import httpx
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from agent import BookChat


def _status_error(error_type, status_code: int, body: dict) -> Exception:
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return error_type(
        "request failed",
        response=response,
        body=body,
    )


class BookChatErrorHandlingTests(unittest.TestCase):
    def _chat_that_raises(self, error: Exception) -> BookChat:
        chat = BookChat.__new__(BookChat)
        chat.agent = Mock()
        chat.agent.invoke.side_effect = error
        chat.thread_id = "test"
        return chat

    def test_expired_or_invalid_api_key_gets_actionable_message(self) -> None:
        error = _status_error(
            AuthenticationError,
            401,
            {"error": {"message": "invalid or expired API key"}},
        )

        with self.assertLogs("agent", level="WARNING"):
            reply = self._chat_that_raises(error).say("Hello", verbose=False)

        self.assertIn("invalid or expired", reply)
        self.assertIn("GROQ_API_KEY", reply)

    def test_context_limit_error_recommends_reset(self) -> None:
        error = _status_error(
            BadRequestError,
            400,
            {"error": {"code": "context_length_exceeded"}},
        )

        with self.assertLogs("agent", level="WARNING"):
            reply = self._chat_that_raises(error).say("Hello", verbose=False)

        self.assertIn("too long", reply)
        self.assertIn("reset", reply)

    def test_rate_limit_error_recommends_waiting(self) -> None:
        error = _status_error(RateLimitError, 429, {"error": {"type": "rate_limit"}})

        with self.assertLogs("agent", level="WARNING"):
            reply = self._chat_that_raises(error).say("Hello", verbose=False)

        self.assertIn("rate-limited", reply)
        self.assertIn("wait", reply)

    def test_other_common_model_failures_are_explained(self) -> None:
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        errors_and_messages = (
            (
                _status_error(NotFoundError, 404, {}),
                "GROQ_MODEL",
            ),
            (
                _status_error(PermissionDeniedError, 403, {}),
                "permissions",
            ),
            (
                APITimeoutError(request=request),
                "too long to respond",
            ),
            (
                APIConnectionError(request=request),
                "internet connection",
            ),
            (
                _status_error(InternalServerError, 500, {}),
                "temporarily unavailable",
            ),
        )
        for error, expected in errors_and_messages:
            with self.subTest(error=type(error).__name__):
                with self.assertLogs("agent", level="WARNING"):
                    reply = self._chat_that_raises(error).say("Hello", verbose=False)
                self.assertIn(expected, reply)


if __name__ == "__main__":
    unittest.main()
