"""widen seasons.name to VARCHAR(30)

Revision ID: e7c9a1b2f3d4
Revises: d4b2c6e8f1a2
Create Date: 2026-08-03

Allarga la colonna `seasons.name` da VARCHAR(10) a VARCHAR(30) per consentire
nomi di stagione più lunghi. Widening di un varchar: sicuro, non perde dati.
"""

import sqlalchemy as sa
from alembic import op

revision = "e7c9a1b2f3d4"
down_revision = "d4b2c6e8f1a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "seasons",
        "name",
        existing_type=sa.String(length=10),
        type_=sa.String(length=30),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "seasons",
        "name",
        existing_type=sa.String(length=30),
        type_=sa.String(length=10),
        existing_nullable=False,
    )
