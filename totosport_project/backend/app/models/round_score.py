import uuid

from sqlalchemy import ForeignKey, SmallInteger, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RoundScore(Base):
    """
    Score aggregato di un giocatore per una giornata, con le componenti tenute
    SEPARATE per poter costruire le 3 classifiche di fine stagione (REGOLAMENTO §6):
    - sign_points        → Classifica Segni
    - exact_points       → (insieme a total_goals) Classifica Pieni+Gol
    - total_goals_points → idem
    - weekend_bonus      → bonus di giornata (0/2/4/6)
    """

    __tablename__ = "round_scores"
    __table_args__ = (UniqueConstraint("player_id", "round_id", name="uq_round_score_player_round"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    round_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rounds.id", ondelete="CASCADE"), nullable=False
    )
    sign_points: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    exact_points: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    total_goals_points: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    weekend_bonus: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    total_round_points: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
