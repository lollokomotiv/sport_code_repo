import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, SmallInteger, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class MatchPrediction(Base):
    """
    Previsione di un giocatore per una partita: sempre risultato esatto (il segno
    1/X/2 è DERIVATO, non salvato — CLAUDE.md §6.1). `points_earned` è popolato
    dallo scoring dopo l'inserimento del risultato.
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
    predicted_home_goals: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    predicted_away_goals: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    points_earned: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
