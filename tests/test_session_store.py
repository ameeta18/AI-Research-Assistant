import unittest
from concurrent.futures import ThreadPoolExecutor

from src.session_store import InMemorySessionStore, SessionCapacityError


class InMemorySessionStoreTests(unittest.TestCase):
    def test_sessions_are_created_independently(self):
        store = InMemorySessionStore(ttl_seconds=60, max_sessions=10)

        session_a = store.get_or_create("session-a")
        session_b = store.get_or_create("session-b")

        self.assertIsNot(session_a, session_b)
        self.assertEqual(len(store), 2)

    def test_same_session_is_created_once_under_concurrency(self):
        store = InMemorySessionStore(ttl_seconds=60, max_sessions=10)

        with ThreadPoolExecutor(max_workers=8) as executor:
            resources = list(
                executor.map(lambda _: store.get_or_create("shared-session"), range(32))
            )

        self.assertEqual(len({id(resource) for resource in resources}), 1)

    def test_expired_sessions_are_removed(self):
        now = [0.0]
        store = InMemorySessionStore(
            ttl_seconds=10,
            max_sessions=10,
            clock=lambda: now[0],
        )
        store.get_or_create("expired-session")

        now[0] = 10.0

        self.assertEqual(store.cleanup_expired(), 1)
        self.assertEqual(len(store), 0)

    def test_capacity_limit_is_enforced(self):
        store = InMemorySessionStore(ttl_seconds=60, max_sessions=1)
        store.get_or_create("first-session")

        with self.assertRaises(SessionCapacityError):
            store.get_or_create("second-session")

    def test_empty_session_id_is_rejected(self):
        store = InMemorySessionStore(ttl_seconds=60, max_sessions=1)

        with self.assertRaisesRegex(ValueError, "session_id"):
            store.get_or_create("  ")


if __name__ == "__main__":
    unittest.main()
