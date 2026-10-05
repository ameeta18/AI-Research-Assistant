import unittest
from unittest.mock import Mock, patch

from src.cache import (
    NoOpJsonCache,
    RedisJsonCache,
    build_cache_key,
    get_cache,
)
from src.config import Settings
from src.reliability import ExternalServiceError
from src.tools.arxiv_tool import _search_arxiv_papers
from src.tools.semantic_scholar_tool import _search_semantic_scholar_papers


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}
        self.deleted = []
        self.read_error = None
        self.write_error = None

    def get(self, key):
        if self.read_error:
            raise self.read_error
        return self.values.get(key)

    def set(self, key, value, ex):
        if self.write_error:
            raise self.write_error
        self.values[key] = value
        self.expirations[key] = ex
        return True

    def delete(self, key):
        self.deleted.append(key)
        self.values.pop(key, None)


class RedisJsonCacheTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.cache = RedisJsonCache(
            self.redis,
            prefix="test-cache",
            ttl_seconds=900,
            recoverable_errors=(ConnectionError,),
        )

    def test_cache_key_is_stable_and_hides_raw_query(self):
        identity = {"query": "private research topic", "max_results": 5}
        first = build_cache_key("app", "papers", identity)
        second = build_cache_key(
            "app",
            "papers",
            {"max_results": 5, "query": "private research topic"},
        )

        self.assertEqual(first, second)
        self.assertNotIn("private research topic", first)
        self.assertNotEqual(
            first,
            build_cache_key("app", "papers", {"query": "different"}),
        )

    def test_json_round_trip_applies_ttl(self):
        identity = {"query": "attention", "max_results": 5}
        papers = [{"title": "Attention Paper"}]

        self.assertTrue(self.cache.set_json("papers", identity, papers))
        self.assertEqual(self.cache.get_json("papers", identity), papers)

        key = build_cache_key("test-cache", "papers", identity)
        self.assertEqual(self.redis.expirations[key], 900)

    def test_invalid_json_is_deleted_and_treated_as_a_miss(self):
        identity = {"query": "attention"}
        key = build_cache_key("test-cache", "papers", identity)
        self.redis.values[key] = "not-json"

        self.assertIsNone(self.cache.get_json("papers", identity))
        self.assertEqual(self.redis.deleted, [key])

    def test_redis_failure_degrades_to_a_cache_miss(self):
        self.redis.read_error = ConnectionError("redis unavailable")

        self.assertIsNone(self.cache.get_json("papers", {"query": "attention"}))

    def test_redis_write_failure_does_not_break_the_caller(self):
        self.redis.write_error = ConnectionError("redis unavailable")

        self.assertFalse(
            self.cache.set_json(
                "papers",
                {"query": "attention"},
                [{"title": "Paper"}],
            )
        )

    def test_disabled_cache_performs_no_work(self):
        cache = NoOpJsonCache()

        self.assertFalse(cache.enabled)
        self.assertIsNone(cache.get_json("papers", {"query": "attention"}))
        self.assertFalse(
            cache.set_json("papers", {"query": "attention"}, [{"title": "Paper"}])
        )

    def test_disabled_configuration_does_not_construct_redis(self):
        settings = Settings(_env_file=None, cache_enabled=False)
        get_cache.cache_clear()

        with (
            patch("src.cache.get_settings", return_value=settings),
            patch("src.cache.RedisJsonCache.from_url") as from_url,
        ):
            cache = get_cache()

        self.assertIsInstance(cache, NoOpJsonCache)
        from_url.assert_not_called()
        get_cache.cache_clear()

    def test_enabled_configuration_constructs_a_lazy_redis_cache(self):
        settings = Settings(
            _env_file=None,
            cache_enabled=True,
            redis_url="redis://localhost:6379/0",
        )
        get_cache.cache_clear()
        expected_cache = Mock()

        with (
            patch("src.cache.get_settings", return_value=settings),
            patch(
                "src.cache.RedisJsonCache.from_url",
                return_value=expected_cache,
            ) as from_url,
        ):
            cache = get_cache()

        self.assertIs(cache, expected_cache)
        from_url.assert_called_once_with("redis://localhost:6379/0")
        get_cache.cache_clear()


class SearchCachingTests(unittest.TestCase):
    def test_arxiv_repeated_search_uses_cached_result(self):
        cache = RedisJsonCache(
            FakeRedis(),
            prefix="test-cache",
            ttl_seconds=900,
        )
        response = Mock()
        response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Attention Paper</title>
            <summary>Research summary</summary>
            <author><name>Researcher</name></author>
            <link href="https://arxiv.org/pdf/1234" type="application/pdf" />
          </entry>
        </feed>"""

        with (
            patch("src.tools.arxiv_tool.get_cache", return_value=cache),
            patch(
                "src.tools.arxiv_tool.select_working_pdf",
                return_value=("https://arxiv.org/pdf/1234", "arXiv"),
            ),
            patch(
                "src.tools.arxiv_tool.request_with_retries",
                return_value=response,
            ) as request,
        ):
            first = _search_arxiv_papers("attention", 5)
            second = _search_arxiv_papers("ATTENTION", 5)

        self.assertEqual(first, second)
        self.assertEqual(request.call_count, 1)

    def test_semantic_scholar_repeated_search_uses_cached_result(self):
        cache = RedisJsonCache(
            FakeRedis(),
            prefix="test-cache",
            ttl_seconds=900,
        )
        response = Mock()
        response.json.return_value = {
            "data": [
                {
                    "title": "Attention Paper",
                    "abstract": "Research summary",
                    "authors": [{"name": "Researcher"}],
                    "year": 2024,
                    "url": "https://example.com/paper",
                    "externalIds": {},
                    "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                }
            ]
        }

        with (
            patch(
                "src.tools.semantic_scholar_tool.get_cache",
                return_value=cache,
            ),
            patch(
                "src.tools.semantic_scholar_tool.select_working_pdf",
                return_value=(
                    "https://example.com/paper.pdf",
                    "Semantic Scholar open access",
                ),
            ),
            patch(
                "src.tools.semantic_scholar_tool.request_with_retries",
                return_value=response,
            ) as request,
        ):
            first = _search_semantic_scholar_papers("attention", 5)
            second = _search_semantic_scholar_papers(" Attention ", 5)

        self.assertEqual(first, second)
        self.assertEqual(request.call_count, 1)

    def test_failed_search_is_not_cached(self):
        cache = Mock()
        cache.get_json.return_value = None
        response = Mock(text="not valid XML")

        with (
            patch("src.tools.arxiv_tool.get_cache", return_value=cache),
            patch(
                "src.tools.arxiv_tool.request_with_retries",
                return_value=response,
            ),
            self.assertRaises(ExternalServiceError),
        ):
            _search_arxiv_papers("attention", 5)

        cache.set_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
