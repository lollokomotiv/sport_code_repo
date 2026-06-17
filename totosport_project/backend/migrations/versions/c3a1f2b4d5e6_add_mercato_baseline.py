"""add mercato_baseline snapshot to table_predictions

Revision ID: c3a1f2b4d5e6
Revises: ab5557b05cee
Create Date: 2026-06-17

Aggiunge la colonna JSONB `mercato_baseline`: snapshot del tabellone originale
catturato alla prima modifica in mercato, per calcolare la penalità rispetto
all'originale (vedi services/tabellone.modify_tabellone).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c3a1f2b4d5e6"
down_revision = "ab5557b05cee"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "table_predictions",
        sa.Column("mercato_baseline", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("table_predictions", "mercato_baseline")
