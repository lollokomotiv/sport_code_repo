import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class SeasonOutcome(Base):
    """
    Risultati reali di una stagione, inseriti dall'admin a fine anno (o man mano).
    Specchia i campi di TablePrediction ed è la base per lo scoring (REGOLAMENTO §2).
    Una sola riga per stagione.

    Note sui campi rispetto allo schema del doc di fase:
    - Aggiunti playoff_b_1..6 e playout_b_1/2 (servono per scorare le voci
      "ai playoff/playout", 20pt ciascuna).
    - I bool *_via_playoff/*_via_playout sono omessi: il metodo di promozione/
      retrocessione è già codificato da quali team stanno in direct_* vs playoff/playout.
    """

    __tablename__ = "season_outcomes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    season_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("seasons.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # ─── Serie A ────────────────────────────────────────────────────────────────
    scudetto_team: Mapped[Optional[str]] = mapped_column(String(80))
    scudetto_points: Mapped[Optional[int]] = mapped_column(Integer)
    relegated_a_1: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_a_2: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_a_3: Mapped[Optional[str]] = mapped_column(String(80))
    top_scorer_a: Mapped[Optional[str]] = mapped_column(String(80))
    top_scorer_a_goals: Mapped[Optional[int]] = mapped_column(Integer)

    # ─── Serie B ────────────────────────────────────────────────────────────────
    promoted_b_direct_1: Mapped[Optional[str]] = mapped_column(String(80))
    promoted_b_direct_2: Mapped[Optional[str]] = mapped_column(String(80))
    promoted_b_first_points: Mapped[Optional[int]] = mapped_column(Integer)
    playoff_b_1: Mapped[Optional[str]] = mapped_column(String(80))
    playoff_b_2: Mapped[Optional[str]] = mapped_column(String(80))
    playoff_b_3: Mapped[Optional[str]] = mapped_column(String(80))
    playoff_b_4: Mapped[Optional[str]] = mapped_column(String(80))
    playoff_b_5: Mapped[Optional[str]] = mapped_column(String(80))
    playoff_b_6: Mapped[Optional[str]] = mapped_column(String(80))
    promoted_b_playoff: Mapped[Optional[str]] = mapped_column(String(80))
    playoffs_held: Mapped[Optional[bool]] = mapped_column(Boolean)
    relegated_b_c_direct_1: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_b_c_direct_2: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_b_c_direct_3: Mapped[Optional[str]] = mapped_column(String(80))
    playout_b_1: Mapped[Optional[str]] = mapped_column(String(80))
    playout_b_2: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_b_c_playout: Mapped[Optional[str]] = mapped_column(String(80))
    playout_held: Mapped[Optional[bool]] = mapped_column(Boolean)
    top_scorer_b: Mapped[Optional[str]] = mapped_column(String(80))
    top_scorer_b_goals: Mapped[Optional[int]] = mapped_column(Integer)

    # ─── Coppe ──────────────────────────────────────────────────────────────────
    coppa_italia_winner: Mapped[Optional[str]] = mapped_column(String(80))
    champions_winner: Mapped[Optional[str]] = mapped_column(String(80))
    europa_winner: Mapped[Optional[str]] = mapped_column(String(80))
    conference_winner: Mapped[Optional[str]] = mapped_column(String(80))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),  # lato Python: evita lazy-load post-UPDATE
        nullable=False,
    )
