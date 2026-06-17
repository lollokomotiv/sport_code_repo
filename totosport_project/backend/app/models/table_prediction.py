import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class TablePrediction(Base):
    """
    Tabellone annuale di un giocatore per una stagione (REGOLAMENTO §2).
    Una sola riga per (player, season). I 6 playoff e i 2 playout sono colonne
    separate perché ognuno è una voce modificabile indipendente (REGOLAMENTO §7.3).
    """

    __tablename__ = "table_predictions"
    __table_args__ = (UniqueConstraint("player_id", "season_id", name="uq_tabellone_player_season"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    season_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )

    # Penalità immediata cumulata dalle modifiche post-mercato (REGOLAMENTO §7.2)
    mercato_penalty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Snapshot del tabellone originale (campo → valore) catturato alla PRIMA
    # modifica in mercato. Serve a calcolare la penalità rispetto all'originale:
    # rimodificare la stessa voce resta -5; tornare al valore originale azzera.
    mercato_baseline: Mapped[Optional[dict]] = mapped_column(JSONB)

    # ─── Serie A ────────────────────────────────────────────────────────────────
    scudetto_team: Mapped[Optional[str]] = mapped_column(String(80))
    scudetto_points_guess: Mapped[Optional[int]] = mapped_column(Integer)
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
    relegated_b_c_direct_1: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_b_c_direct_2: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_b_c_direct_3: Mapped[Optional[str]] = mapped_column(String(80))
    playout_b_1: Mapped[Optional[str]] = mapped_column(String(80))
    playout_b_2: Mapped[Optional[str]] = mapped_column(String(80))
    relegated_b_c_playout: Mapped[Optional[str]] = mapped_column(String(80))
    top_scorer_b: Mapped[Optional[str]] = mapped_column(String(80))
    top_scorer_b_goals: Mapped[Optional[int]] = mapped_column(Integer)

    # ─── Coppe ──────────────────────────────────────────────────────────────────
    coppa_italia_winner: Mapped[Optional[str]] = mapped_column(String(80))
    champions_winner: Mapped[Optional[str]] = mapped_column(String(80))
    europa_winner: Mapped[Optional[str]] = mapped_column(String(80))
    conference_winner: Mapped[Optional[str]] = mapped_column(String(80))

    # ─── Scoring (popolati da score_tabellone) ───────────────────────────────────
    total_points: Mapped[Optional[int]] = mapped_column(Integer)
    points_breakdown: Mapped[Optional[dict]] = mapped_column(JSONB)
    scored_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),  # lato Python: evita lazy-load post-UPDATE
        nullable=False,
    )
