import enum
import uuid
from typing import Optional

from sqlalchemy import DateTime, Enum, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class SeasonStatus(str, enum.Enum):
    setup = "setup"
    active = "active"
    mercato = "mercato"
    closed = "closed"


class Season(Base):
    __tablename__ = "seasons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(String(10), nullable=False)  # es. "2025-26"
    status: Mapped[SeasonStatus] = mapped_column(
        Enum(SeasonStatus, name="seasonstatus"),
        nullable=False,
        default=SeasonStatus.setup,
    )
    tabellone_deadline: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    modification_deadline: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
