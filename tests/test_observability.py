import json
import logging
from contextlib import nullcontext
import unittest
import uuid
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.config import Settings
from src.observability import (
    JsonFormatter,
    configure_logging,
    get_langsmith_client,
    log_context,
    trace_research_run,
)


class StructuredLoggingTests(unittest.TestCase):
    def test_json_logs_include_correlation_and_operational_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="ai_researcher.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="operation_completed",
            args=(),
            exc_info=None,
        )
        record.event_data = {"duration_ms": 12.5, "status_code": 200}

        with log_context(request_id="request-1", session_id="session-1"):
            payload = json.loads(formatter.format(record))

        self.assertEqual(payload["event"], "operation_completed")
        self.assertEqual(payload["request_id"], "request-1")
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["duration_ms"], 12.5)
        self.assertEqual(payload["status_code"], 200)

    def test_logging_configuration_is_idempotent(self):
        settings = Settings(_env_file=None, log_level="INFO")
        logger = configure_logging(settings)
        initial_handlers = list(logger.handlers)

        configure_logging(settings)

        self.assertEqual(logger.handlers, initial_handlers)

    def test_chat_logs_do_not_include_prompt_content(self):
        from src.api import ChatRequest, chat

        graph = Mock()
        graph.stream.return_value = [
            {"messages": [AIMessage(content="safe response")]},
        ]
        logger = Mock()
        secret_prompt = "private unpublished research idea"

        with (
            patch("src.api.get_graph", return_value=graph),
            patch("src.api.logger", logger),
            patch("src.api.trace_research_run", return_value=nullcontext()),
        ):
            result = chat(
                ChatRequest(
                    message=secret_prompt,
                    session_id=uuid.uuid4(),
                )
            )

        self.assertEqual(result.response, "safe response")
        self.assertNotIn(secret_prompt, repr(logger.method_calls))

    def test_http_responses_include_a_request_id(self):
        from src.api import app

        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        uuid.UUID(response.headers["X-Request-ID"])


class LangSmithTracingTests(unittest.TestCase):
    def tearDown(self):
        get_langsmith_client.cache_clear()

    def test_disabled_tracing_does_not_create_a_client(self):
        settings = Settings(_env_file=None, langsmith_tracing=False)
        get_langsmith_client.cache_clear()

        with (
            patch("src.observability.get_settings", return_value=settings),
            patch("src.observability.Client") as client,
        ):
            self.assertIsNone(get_langsmith_client())

        client.assert_not_called()

    def test_enabled_tracing_uses_private_capture_defaults(self):
        settings = Settings(
            _env_file=None,
            langsmith_tracing=True,
            langsmith_api_key="trace-secret",
        )
        get_langsmith_client.cache_clear()

        with (
            patch("src.observability.get_settings", return_value=settings),
            patch("src.observability.Client") as client,
        ):
            get_langsmith_client()

        client.assert_called_once_with(
            api_key="trace-secret",
            hide_inputs=True,
            hide_outputs=True,
            tracing_sampling_rate=1.0,
        )

    def test_trace_context_receives_only_correlation_metadata(self):
        settings = Settings(_env_file=None, langsmith_tracing=False)
        get_langsmith_client.cache_clear()

        with (
            patch("src.observability.get_settings", return_value=settings),
            patch(
                "src.observability.tracing_context",
                return_value=nullcontext(),
            ) as tracing,
            log_context(request_id="request-1"),
        ):
            with trace_research_run(interface="api", session_id="session-1"):
                pass

        kwargs = tracing.call_args.kwargs
        self.assertFalse(kwargs["enabled"])
        self.assertEqual(kwargs["metadata"]["request_id"], "request-1")
        self.assertNotIn("prompt", kwargs["metadata"])


if __name__ == "__main__":
    unittest.main()
