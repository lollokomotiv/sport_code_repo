import ssl
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# In produzione (Neon o altro Postgres gestito) serve SSL. `statement_cache_size=0`
# rende la connessione compatibile anche con l'endpoint "pooled" di Neon (PgBouncer),
# che non gestisce i prepared statement di asyncpg.
_connect_args: dict = {}
if settings.db_ssl:
    _connect_args = {"ssl": ssl.create_default_context(), "statement_cache_size": 0}

engine = create_async_engine(
    settings.database_url,
    echo=False,  # metti True per loggare tutte le query SQL in dev
    pool_pre_ping=True,  # controlla la connessione prima di usarla
    pool_recycle=300,  # ricicla le connessioni (Neon chiude quelle inattive)
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(DeclarativeBase):
    """Base class per tutti i modelli ORM."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: fornisce una sessione DB per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
