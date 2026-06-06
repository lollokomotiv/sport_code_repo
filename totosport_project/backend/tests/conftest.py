"""
Fixture condivise per i test di integrazione.

Strategia:
- DB di test dedicato `totosport_test` nello stesso container Postgres, creato
  UNA volta a inizio sessione (fixture sync → nessun event loop coinvolto).
- Engine async + schema (`drop_all`/`create_all`) ricreati PER TEST: così ogni
  test gira sullo stesso event loop delle proprie fixture, evitando l'errore
  asyncpg "another operation in progress" causato dal mix di loop diversi.
- `get_db` è sovrascritto con una sessione nuova per request (come in prod).
"""

import psycopg2
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models  # noqa: F401  — registra i modelli su Base.metadata
from app.config import settings
from app.database import Base, get_db
from app.main import app

TEST_DB_NAME = "totosport_test"

_base_url = make_url(settings.database_url)
_test_url = _base_url.set(database=TEST_DB_NAME)


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    """Crea il database di test una sola volta (connessione sync in autocommit)."""
    conn = psycopg2.connect(
        host=_base_url.host,
        port=_base_url.port,
        user=_base_url.username,
        password=_base_url.password,
        dbname="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}"')
            cur.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        conn.close()
    yield


@pytest_asyncio.fixture
async def engine(_test_database):
    eng = create_async_engine(_test_url.render_as_string(hide_password=False))
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def db_session(session_factory) -> AsyncSession:
    """Sessione per seedare dati di test. Il seed deve fare commit esplicito."""
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(session_factory) -> AsyncClient:
    """Client HTTP con get_db sovrascritto: una sessione nuova per ogni request."""

    async def _override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
