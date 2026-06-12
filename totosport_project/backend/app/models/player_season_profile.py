import uuid

from sqlalchemy import ForeignKey, SmallInteger, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PlayerSeasonProfile(Base):
    """
    Dati di un giocatore a livello di stagione che non hanno casa altrove:
    i 3 bonus di fine stagione (0 o 10 ciascuno, REGOLAMENTO §6).
    I punti tabellone e la penalità mercato vivono già su TablePrediction.
    """

    __tablename__ = "player_season_profiles"
    __table_args__ = (
        UniqueConstraint("player_id", "season_id", name="uq_profile_player_season"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    season_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )
    season_bonus_signs: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    season_bonus_exacts: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    season_bonus_tabellone: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
