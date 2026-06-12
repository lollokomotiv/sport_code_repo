import uuid

from pydantic import BaseModel


class LeaderboardEntry(BaseModel):
    """Riga della classifica generale di stagione."""

    rank: int
    player_id: uuid.UUID
    username: str
    total_points: int
    sign_points: int
    exact_points: int
    total_goals_points: int
    weekend_bonus_total: int
    tabellone_points: int
    season_bonus_total: int  # somma dei 3 bonus di fine stagione (0..30)


class RoundLeaderboardEntry(BaseModel):
    """Riga della classifica di una singola giornata."""

    rank: int
    player_id: uuid.UUID
    username: str
    round_points: int
    sign_points: int
    exact_points: int
    total_goals_points: int
    weekend_bonus: int


class MetricLeaderboardEntry(BaseModel):
    """Classifica per singola metrica (segni / pieni+gol / tabellone)."""

    rank: int
    player_id: uuid.UUID
    username: str
    points: int


class SeasonBonusResult(BaseModel):
    """Esito della finalizzazione: chi ha preso i 3 bonus."""

    bonus_signs: list[uuid.UUID]
    bonus_exacts: list[uuid.UUID]
    bonus_tabellone: list[uuid.UUID]
