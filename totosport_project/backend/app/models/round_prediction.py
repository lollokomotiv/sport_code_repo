import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, SmallInteger, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base
from app.models.round import Competition


class RoundPrediction(Base):
    """
    Previsione del TOTALE GOL di una giornata, separata PER LEGA: in una giornata
    combinata (Serie A + Serie B) il giocatore inserisce due righe, una per
    `serie_a` e una per `serie_b`. Solo Serie A/B hanno il totale gol.
    """

    __tablename__ = "round_predictions"
    __table_args__ = (
        UniqueConstraint("player_id", "round_id", "competition", name="uq_round_pred_player_round_comp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    round_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rounds.id", ondelete="CASCADE"), nullable=False
    )
    competition: Mapped[Competition] = mapped_column(
        Enum(Competition, name="competition"), nullable=False
    )
    total_goals_guess: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    points_earned: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
