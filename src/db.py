from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import settings
from src.db_models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _prepare_sqlite_path(url: str) -> None:
    if not url.startswith("sqlite:///") or url.endswith(":memory:"):
        return
    raw_path = url.removeprefix("sqlite:///")
    Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def build_engine(url: str | None = None) -> Engine:
    db_url = url or settings.database_url
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if db_url.endswith(":memory:"):
            kwargs["poolclass"] = StaticPool
        else:
            _prepare_sqlite_path(db_url)
    return create_engine(db_url, **kwargs)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    Base.metadata.create_all(bind=engine or get_engine())


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def reset_engine(url: str | None = None) -> Engine:
    """Used by tests and by the CLI when DATABASE_URL changes."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = build_engine(url)
    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine
