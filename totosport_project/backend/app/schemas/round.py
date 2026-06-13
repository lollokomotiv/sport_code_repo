import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.round import Competition, RoundStatus


# ─── Match ───────────────────────────────────────────────────────────────────


class MatchCreate(BaseModel):
    """Aggiunta partita: manuale (home/away/competition/kickoff) o da staged fixture."""

    home_team: Optional[str] = None
    away_team: Optional[str] = None
    competition: Optional[Competition] = None  # obbligatoria per inserimento manuale
    requires_exact_score: bool = False  # True → si pronostica anche il risultato esatto
    kickoff: Optional[datetime] = None
    staged_fixture_id: Optional[uuid.UUID] = None


class MatchResultUpdate(BaseModel):
    home_goals: int
    away_goals: int


class MatchOut(BaseModel):
    id: uuid.UUID
    round_id: uuid.UUID
    competition: Competition
    home_team: str
    away_team: str
    requires_exact_score: bool
    kickoff: Optional[datetime]
    actual_home_goals: Optional[int]
    actual_away_goals: Optional[int]
    api_fixture_id: Optional[int]

    model_config = {"from_attributes": True}


# ─── Round ───────────────────────────────────────────────────────────────────


class RoundCreate(BaseModel):
    name: str
    competition: Competition
    matchday: Optional[int] = None
    deadline: Optional[datetime] = None


class RoundStatusUpdate(BaseModel):
    status: RoundStatus


class RoundOut(BaseModel):
    id: uuid.UUID
    season_id: uuid.UUID
    name: str
    competition: Competition
    matchday: Optional[int]
    deadline: Optional[datetime]
    status: RoundStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class RoundDetailOut(RoundOut):
    """Dettaglio giornata con le sue partite."""

    matches: list[MatchOut] = []


# ─── Staged Fixture ──────────────────────────────────────────────────────────


class StagedFixtureOut(BaseModel):
    id: uuid.UUID
    api_fixture_id: int
    competition: Competition
    season: Optional[str]
    matchday: Optional[int]
    home_team: str
    away_team: str
    kickoff: Optional[datetime]
    added_to_round: bool

    model_config = {"from_attributes": True}
