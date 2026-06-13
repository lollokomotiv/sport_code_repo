"""match exact-score flag and explicit sign

Revision ID: ab5557b05cee
Revises: 7ea48c4fe03d
Create Date: 2026-06-13 14:50:31.764082

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ab5557b05cee'
down_revision: Union[str, Sequence[str], None] = '7ea48c4fe03d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # matches: flag "richiede risultato esatto" (default false → solo segno)
    op.add_column(
        "matches",
        sa.Column("requires_exact_score", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )

    # match_predictions: segno esplicito + goal ora opzionali
    op.add_column("match_predictions", sa.Column("predicted_sign", sa.String(length=1), nullable=True))
    # Backfill del segno dalle previsioni esistenti (avevano sempre i goal)
    op.execute(
        """
        UPDATE match_predictions SET predicted_sign = CASE
            WHEN predicted_home_goals > predicted_away_goals THEN '1'
            WHEN predicted_home_goals = predicted_away_goals THEN 'X'
            ELSE '2' END
        WHERE predicted_sign IS NULL
        """
    )
    op.alter_column("match_predictions", "predicted_sign", nullable=False)
    op.alter_column("match_predictions", "predicted_home_goals", nullable=True)
    op.alter_column("match_predictions", "predicted_away_goals", nullable=True)


def downgrade() -> None:
    op.alter_column("match_predictions", "predicted_away_goals", nullable=False)
    op.alter_column("match_predictions", "predicted_home_goals", nullable=False)
    op.drop_column("match_predictions", "predicted_sign")
    op.drop_column("matches", "requires_exact_score")
