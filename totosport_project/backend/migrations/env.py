import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

# Carica .env per avere DATABASE_URL disponibile
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Importa Base con tutti i modelli per il supporto --autogenerate
from app.database import Base  # noqa: E402
import app.models  # noqa: F401, E402

target_metadata = Base.metadata


def get_url() -> str:
    """
    Legge DATABASE_URL dal .env e converte asyncpg → psycopg2.
    Alembic usa sempre un engine sincrono.

    Se DB_SSL è attivo (produzione, es. Neon), aggiunge `sslmode=require` alla
    URL psycopg2 (asyncpg gestisce l'SSL nell'engine dell'app; psycopg2 lo vuole
    nella URL). Così migrazioni e runtime usano la stessa DATABASE_URL asyncpg.
    """
    url = os.getenv("DATABASE_URL", "").replace(
        "postgresql+asyncpg://", "postgresql+psycopg2://"
    )
    db_ssl = os.getenv("DB_SSL", "").lower() in ("1", "true", "yes")
    if db_ssl and "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
