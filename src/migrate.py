"""Run application migrations and initialize LangGraph checkpoint tables."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from langgraph.checkpoint.postgres import PostgresSaver

from src.config import get_settings


def _checkpoint_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def migrate() -> None:
    settings = get_settings()
    database_url = settings.get_database_url()
    if not settings.database_enabled or database_url is None:
        raise RuntimeError("Set DATABASE_ENABLED=true and DATABASE_URL before migrating")

    root = Path(__file__).resolve().parents[1]
    alembic_config = Config(str(root / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(alembic_config, "head")

    with PostgresSaver.from_conn_string(_checkpoint_url(database_url)) as checkpointer:
        checkpointer.setup()


if __name__ == "__main__":
    migrate()
