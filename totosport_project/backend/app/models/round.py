import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, SmallInteger, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class Competition(str, enum.Enum):
    serie_a = "serie_a"
    serie_b = "serie_b"
    champions_league = "champions_league"
    mixed = "mixed"


class RoundStatus(str, enum.Enum):
    draft = "draft"
    open = "open"
    closed = "closed"
    completed = "completed"


class Round(Base):
    """Giornata: aggrega partite di una o più competizioni nello stesso weekend."""

    __tablename__ = "rounds"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    season_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    competition: Mapped[Competition] = mapped_column(
        Enum(Competition, name="competition"), nullable=False
    )
    matchday: Mapped[Optional[int]] = mapped_column(SmallInteger)
    deadline: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[RoundStatus] = mapped_column(
        Enum(RoundStatus, name="roundstatus"),
        nullable=False,
        server_default=RoundStatus.draft.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
