import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from src.config import Settings, get_llm, get_settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_valid_without_secrets(self):
        settings = Settings(_env_file=None)

        self.assertEqual(settings.llm_model, "gemini-2.5-flash")
        self.assertEqual(settings.chunk_size, 1000)
        self.assertEqual(settings.session_ttl_minutes, 60)
        self.assertEqual(settings.max_active_sessions, 100)
        self.assertFalse(settings.cache_enabled)
        self.assertFalse(settings.database_enabled)
        self.assertEqual(settings.cache_ttl_seconds, 900)
        self.assertIsNone(settings.google_api_key)

    def test_google_key_is_required_only_when_gemini_is_used(self):
        settings = Settings(_env_file=None)

        with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
            settings.require_google_api_key()

    def test_session_key_override_builds_llm_without_changing_settings(self):
        with patch("src.config.ChatGoogleGenerativeAI") as chat_model:
            get_llm("session-only-secret")

        self.assertEqual(
            chat_model.call_args.kwargs["google_api_key"],
            "session-only-secret",
        )
        self.assertNotIn("session-only-secret", repr(get_settings()))

    def test_secret_values_are_redacted_from_settings_repr(self):
        settings = Settings(
            _env_file=None,
            google_api_key="test-secret-value",
            langsmith_api_key="trace-secret-value",
            redis_url="redis://user:redis-secret@localhost:6379/0",
            database_url="postgresql+psycopg://user:database-secret@localhost/app",
        )

        self.assertNotIn("test-secret-value", repr(settings))
        self.assertNotIn("trace-secret-value", repr(settings))
        self.assertNotIn("redis-secret", repr(settings))
        self.assertNotIn("database-secret", repr(settings))
        self.assertEqual(settings.require_google_api_key(), "test-secret-value")

    def test_invalid_chunk_configuration_is_rejected(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, chunk_size=100, chunk_overlap=100)

    def test_invalid_session_limits_are_rejected(self):
        for invalid_setting in (
            {"session_ttl_minutes": 0},
            {"max_active_sessions": 0},
        ):
            with self.subTest(invalid_setting=invalid_setting):
                with self.assertRaises(ValidationError):
                    Settings(_env_file=None, **invalid_setting)

    def test_invalid_reliability_settings_are_rejected(self):
        for invalid_setting in (
            {"http_max_attempts": 0},
            {"http_jitter_seconds": -1},
            {"llm_max_retries": -1},
            {"max_pdf_bytes": 0},
            {"arxiv_max_results": 11},
            {"langsmith_tracing_sampling_rate": 1.1},
            {"langsmith_tracing": True, "langsmith_api_key": None},
            {"cache_ttl_seconds": 0},
            {"cache_key_prefix": "   "},
            {"cache_enabled": True, "redis_url": None},
            {"database_enabled": True, "database_url": None},
            {"database_max_overflow": -1},
        ):
            with self.subTest(invalid_setting=invalid_setting):
                with self.assertRaises(ValidationError):
                    Settings(_env_file=None, **invalid_setting)

    def test_cached_settings_are_process_wide(self):
        self.assertIs(get_settings(), get_settings())


class ApiContractTests(unittest.TestCase):
    def test_chat_request_does_not_accept_api_keys(self):
        from src.api import ChatRequest

        self.assertNotIn("api_key", ChatRequest.model_fields)
        with self.assertRaises(ValidationError):
            ChatRequest(message="hello", api_key="client-secret")

    def test_chat_message_must_be_non_blank_and_bounded(self):
        from src.api import ChatRequest

        with self.assertRaises(ValidationError):
            ChatRequest(message="   ")
        with self.assertRaises(ValidationError):
            ChatRequest(message="x" * 10_001)

    def test_unexpected_api_errors_do_not_leak_internal_details(self):
        from src.api import ChatRequest, chat

        with (
            patch("src.api.get_graph", side_effect=RuntimeError("secret detail")),
            self.assertRaises(HTTPException) as raised,
        ):
            chat(ChatRequest(message="find a paper"))

        self.assertEqual(raised.exception.status_code, 500)
        self.assertNotIn("secret detail", raised.exception.detail)


class StreamlitCredentialPolicyTests(unittest.TestCase):
    def test_streamlit_never_falls_back_to_server_google_key(self):
        app_source = (
            Path(__file__).resolve().parent.parent / "src" / "app.py"
        ).read_text(encoding="utf-8")

        self.assertIn('effective_api_key = ui_api_key.strip()', app_source)
        self.assertNotIn("require_google_api_key", app_source)
        self.assertNotIn("settings.google_api_key", app_source)
        self.assertNotIn("settings.app_env", app_source)


if __name__ == "__main__":
    unittest.main()
