import os
import unittest
from uuid import uuid4

from src.checkpointing import _checkpoint_url, get_checkpointer
from src.config import Settings
from src.persistence import NoOpPersistence, PostgresPersistence
from src.persistence.schema import metadata, sessions


class PersistenceConfigurationTests(unittest.TestCase):
    def test_postgres_is_opt_in(self):
        settings = Settings(_env_file=None)

        self.assertFalse(settings.database_enabled)
        self.assertIsNone(settings.database_url)
        self.assertIsInstance(get_checkpointer(), object)

    def test_enabling_postgres_requires_a_postgres_url(self):
        with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
            Settings(_env_file=None, database_enabled=True)

        with self.assertRaisesRegex(ValueError, "PostgreSQL"):
            Settings(
                _env_file=None,
                database_enabled=True,
                database_url="sqlite:///local.db",
            )

    def test_database_secret_is_redacted(self):
        settings = Settings(
            _env_file=None,
            database_url="postgresql+psycopg://user:secret-password@localhost/app",
        )

        self.assertNotIn("secret-password", repr(settings))

    def test_checkpoint_url_removes_only_sqlalchemy_driver_marker(self):
        self.assertEqual(
            _checkpoint_url("postgresql+psycopg://user:pass@postgres/app"),
            "postgresql://user:pass@postgres/app",
        )

    def test_auth_ready_session_owner_is_nullable(self):
        self.assertTrue(sessions.c.user_id.nullable)
        self.assertEqual(
            set(metadata.tables),
            {
                "users",
                "sessions",
                "messages",
                "papers",
                "session_papers",
                "paper_chunks",
                "feedback",
            },
        )


class NoOpPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.store = NoOpPersistence()
        self.session_id = str(uuid4())

    def test_disabled_store_has_safe_empty_behavior(self):
        self.store.ensure_session(self.session_id)
        self.assertIsNone(self.store.save_message(self.session_id, "user", "hello"))
        self.assertEqual(self.store.list_messages(self.session_id), [])
        self.assertTrue(self.store.healthcheck())

    def test_invalid_identifiers_and_feedback_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid UUID"):
            self.store.ensure_session("not-a-uuid")
        with self.assertRaisesRegex(ValueError, "rating"):
            self.store.save_feedback(self.session_id, 0)


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL"),
    "TEST_DATABASE_URL is required for PostgreSQL integration tests",
)
class PostgresPersistenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = PostgresPersistence(os.environ["TEST_DATABASE_URL"])

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_messages_chunks_feedback_and_healthcheck(self):
        session_id = str(uuid4())
        self.store.ensure_session(session_id)

        user_message_id = self.store.save_message(session_id, "user", "question")
        assistant_message_id = self.store.save_message(
            session_id, "assistant", "answer"
        )
        stored_messages = self.store.list_messages(session_id)

        self.assertEqual([message.content for message in stored_messages], ["question", "answer"])
        self.assertEqual(user_message_id + 1, assistant_message_id)

        paper_id = self.store.save_paper_chunks(
            session_id,
            "A durable paper",
            "https://arxiv.org/pdf/1234.5678",
            ["first chunk", "second chunk"],
        )
        self.assertIsNotNone(paper_id)
        stored_chunks = self.store.load_paper_chunks(session_id)
        self.assertEqual([chunk.content for chunk in stored_chunks], ["first chunk", "second chunk"])

        feedback_id = self.store.save_feedback(
            session_id,
            1,
            message_id=assistant_message_id,
            comment="helpful",
        )
        self.assertIsNotNone(feedback_id)
        self.assertTrue(self.store.healthcheck())


if __name__ == "__main__":
    unittest.main()
