"""add_user_season

Revision ID: d9b66a27204f
Revises:
Create Date: 2026-06-06 15:41:11.536482

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d9b66a27204f"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # gen_random_uuid() è nativo in Postgres 13+, ma pgcrypto è innocuo e
    # garantisce la funzione anche su immagini più vecchie.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # Gli ENUM vengono creati automaticamente dalla colonna sa.Enum di ciascuna
    # tabella (ogni tipo è usato da una sola tabella, quindi nessun doppione).
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(100), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("admin", "player", name="userrole"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("email"),
    )

    op.create_table(
        "seasons",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(10), nullable=False),
        sa.Column(
            "status",
            sa.Enum("setup", "active", "mercato", "closed", name="seasonstatus"),
            nullable=False,
            server_default="setup",
        ),
        sa.Column("tabellone_deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("modification_deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("seasons")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS seasonstatus")
    op.execute("DROP TYPE IF EXISTS userrole")
