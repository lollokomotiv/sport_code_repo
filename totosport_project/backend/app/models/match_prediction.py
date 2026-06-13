import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, SmallInteger, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class MatchPrediction(Base):
    """
    Previsione di un giocatore per una partita.

    `predicted_sign` (1/X/2) è SCELTO esplicitamente dal giocatore ed è presente
    su ogni partita. Il risultato esatto (`predicted_home_goals`/`away_goals`) è
    opzionale: si compila solo sulle partite con `requires_exact_score = True`.
    `points_earned` è popolato dallo scoring dopo l'inserimento del risultato.
    """

    __tablename__ = "match_predictions"
    __table_args__ = (UniqueConstraint("player_id", "match_id", name="uq_match_pred_player_match"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"), nullable=False
    )
    predicted_sign: Mapped[str] = mapped_column(String(1), nullable=False)  # '1' | 'X' | '2'
    predicted_home_goals: Mapped[Optional[int]] = mapped_column(SmallInteger)
    predicted_away_goals: Mapped[Optional[int]] = mapped_column(SmallInteger)
    points_earned: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
