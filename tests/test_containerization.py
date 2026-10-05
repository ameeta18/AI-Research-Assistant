import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ContainerConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        cls.compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
        cls.dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(
            encoding="utf-8"
        )

    def test_runtime_is_non_root_and_has_a_healthcheck(self):
        self.assertIn("USER app", self.dockerfile)
        self.assertIn("HEALTHCHECK", self.dockerfile)
        self.assertIn("/_stcore/health", self.dockerfile)

    def test_dependencies_and_tectonic_are_pinned(self):
        self.assertIn("uv sync --locked", self.dockerfile)
        self.assertIn("ARG TECTONIC_VERSION=0.17.0", self.dockerfile)
        self.assertIn("sha256sum --check --strict", self.dockerfile)

    def test_compose_does_not_expose_the_owner_gemini_key(self):
        self.assertNotIn("GOOGLE_API_KEY", self.compose)
        self.assertNotIn("env_file:", self.compose)

    def test_compose_uses_private_healthy_redis(self):
        self.assertIn("REDIS_URL: redis://redis:6379/0", self.compose)
        self.assertIn("condition: service_healthy", self.compose)
        redis_section = self.compose.split("  redis:", maxsplit=1)[1]
        self.assertIn("user: redis", redis_section)
        self.assertNotIn("ports:", redis_section)

    def test_compose_uses_private_postgres_and_migrations(self):
        self.assertIn("image: postgres:17-alpine", self.compose)
        self.assertIn("command: [python, -m, src.migrate]", self.compose)
        self.assertIn("condition: service_completed_successfully", self.compose)
        postgres_section = self.compose.split("  postgres:", maxsplit=1)[1].split(
            "  redis:", maxsplit=1
        )[0]
        self.assertIn("pg_isready", postgres_section)
        self.assertNotIn("ports:", postgres_section)

    def test_database_migrations_are_in_the_runtime_image(self):
        self.assertIn("COPY --chown=app:app migrations ./migrations", self.dockerfile)
        self.assertIn("COPY --chown=app:app alembic.ini ./alembic.ini", self.dockerfile)

    def test_secrets_and_local_artifacts_are_excluded_from_build_context(self):
        ignored = set(self.dockerignore.splitlines())
        self.assertIn(".env", ignored)
        self.assertIn(".venv", ignored)
        self.assertIn("output/", ignored)
        self.assertIn("tectonic.exe", ignored)


if __name__ == "__main__":
    unittest.main()
