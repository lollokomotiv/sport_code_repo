"""add competition to match

Revision ID: f74a6808c718
Revises: 726952479686
Create Date: 2026-06-12 12:49:20.475015

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f74a6808c718'
down_revision: Union[str, Sequence[str], None] = '726952479686'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Riusa il tipo enum 'competition' già creato dalla migrazione di rounds (create_type=False)
competition_enum = postgresql.ENUM(
    "serie_a", "serie_b", "champions_league", "mixed", name="competition", create_type=False
)


def upgrade() -> None:
    # Aggiunge la colonna con un default temporaneo per backfillare le righe esistenti,
    # poi rimuove il default (ogni partita riceve la competizione esplicitamente dal codice).
    op.add_column(
        "matches",
        sa.Column("competition", competition_enum, nullable=False, server_default="serie_a"),
    )
    op.alter_column("matches", "competition", server_default=None)


def downgrade() -> None:
    op.drop_column("matches", "competition")
