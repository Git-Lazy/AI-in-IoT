"""Engine, session factory, and first-run schema creation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import PROJECT_ROOT, settings
from app.db.models import Base

if settings.database_url.startswith("sqlite"):
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)

engine = create_engine(
    settings.database_url,
    # SQLite only: the vision thread and the API thread share this engine.
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create tables if they don't exist. Alembic takes over once the schema settles."""
    Base.metadata.create_all(engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Add columns introduced after a table was first created.

    `create_all` makes new tables but never alters existing ones, so a database
    from an earlier version would be missing new columns. Until Alembic is set
    up, add them here with their model default so existing rows stay valid.
    """
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                default = column.default.arg if column.default is not None else None
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column.type.compile(engine.dialect)}"
                if isinstance(default, str):
                    ddl += f" DEFAULT '{default}'"
                elif isinstance(default, bool):
                    ddl += f" DEFAULT {int(default)}"
                conn.execute(text(ddl))


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background workers."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
