import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class TablePredictionModification(Base):
    """
    Log di una singola modifica post-mercato a una voce del tabellone
    (REGOLAMENTO §7). La presenza di una riga per un `field_name` segnala allo
    scoring finale che quel campo va cappato al 50% dei punti normali.
    """

    __tablename__ = "table_prediction_modifications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("table_predictions.id", ondelete="CASCADE"),
        nullable=False,
    )
    field_name: Mapped[str] = mapped_column(String(80), nullable=False)
    old_value: Mapped[Optional[str]] = mapped_column(String)
    new_value: Mapped[Optional[str]] = mapped_column(String)
    penalty_points: Mapped[int] = mapped_column(Integer, nullable=False, default=-5)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
