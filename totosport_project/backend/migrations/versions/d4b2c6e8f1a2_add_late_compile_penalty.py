"""add late_compile_penalty to table_predictions

Revision ID: d4b2c6e8f1a2
Revises: c3a1f2b4d5e6
Create Date: 2026-06-17

Penalità una tantum (-30) per chi compila il tabellone in ritardo, durante il
mercato. Tenuta separata da mercato_penalty per non confonderla con i -5 delle
modifiche voce-per-voce.
"""

import sqlalchemy as sa
from alembic import op

revision = "d4b2c6e8f1a2"
down_revision = "c3a1f2b4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "table_predictions",
        sa.Column("late_compile_penalty", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("table_predictions", "late_compile_penalty")
