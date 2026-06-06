import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, Integer, SmallInteger, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base
from app.models.round import Competition


class StagedFixture(Base):
    """
    Cache di una fixture recuperata da API-Football (Fase 7). L'admin la può
    aggiungere a una giornata; `added_to_round` evita di riusarla due volte.
    """

    __tablename__ = "staged_fixtures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    api_fixture_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    competition: Mapped[Competition] = mapped_column(
        Enum(Competition, name="competition"), nullable=False
    )
    season: Mapped[Optional[str]] = mapped_column(String(10))
    matchday: Mapped[Optional[int]] = mapped_column(SmallInteger)
    home_team: Mapped[str] = mapped_column(String(80), nullable=False)
    away_team: Mapped[str] = mapped_column(String(80), nullable=False)
    kickoff: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    added_to_round: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
