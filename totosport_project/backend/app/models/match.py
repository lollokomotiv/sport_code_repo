import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, SmallInteger, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base
from app.models.round import Competition


class Match(Base):
    """
    Partita di una giornata. Richiede sempre un risultato esatto (nessun
    `prediction_type`, vedi CLAUDE.md §6.1). `api_fixture_id` collega alla
    fixture di API-Football per l'auto-sync (Fase 7); NULL se inserita a mano.
    """

    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    round_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rounds.id", ondelete="CASCADE"), nullable=False
    )
    # Lega della singola partita: in una giornata 'mixed' serve per sommare i gol
    # per lega (totale gol di Serie A vs Serie B).
    competition: Mapped[Competition] = mapped_column(
        Enum(Competition, name="competition"), nullable=False
    )
    home_team: Mapped[str] = mapped_column(String(80), nullable=False)
    away_team: Mapped[str] = mapped_column(String(80), nullable=False)
    kickoff: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    actual_home_goals: Mapped[Optional[int]] = mapped_column(SmallInteger)
    actual_away_goals: Mapped[Optional[int]] = mapped_column(SmallInteger)
    api_fixture_id: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
