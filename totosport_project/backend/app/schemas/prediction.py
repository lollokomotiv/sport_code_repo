import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, computed_field, field_validator

from app.models.round import Competition
from app.services.scoring import derive_sign


# ─── Match prediction ─────────────────────────────────────────────────────────


class MatchPredictionCreate(BaseModel):
    match_id: uuid.UUID
    predicted_home_goals: int
    predicted_away_goals: int

    @field_validator("predicted_home_goals", "predicted_away_goals")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("I gol non possono essere negativi")
        return v


class MatchPredictionOut(BaseModel):
    id: uuid.UUID
    player_id: uuid.UUID
    match_id: uuid.UUID
    predicted_home_goals: int
    predicted_away_goals: int
    points_earned: int
    submitted_at: Optional[datetime]

    model_config = {"from_attributes": True}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def derived_sign(self) -> str:
        return derive_sign(self.predicted_home_goals, self.predicted_away_goals)


# ─── Round (total goals) prediction ───────────────────────────────────────────


class RoundPredictionCreate(BaseModel):
    round_id: uuid.UUID
    competition: Competition
    total_goals_guess: int

    @field_validator("competition")
    @classmethod
    def only_serie(cls, v: Competition) -> Competition:
        if v not in (Competition.serie_a, Competition.serie_b):
            raise ValueError("Il totale gol si prevede solo per serie_a o serie_b")
        return v

    @field_validator("total_goals_guess")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Il totale gol non può essere negativo")
        return v


class RoundPredictionOut(BaseModel):
    id: uuid.UUID
    player_id: uuid.UUID
    round_id: uuid.UUID
    competition: Competition
    total_goals_guess: int
    points_earned: int
    submitted_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ─── Round score ──────────────────────────────────────────────────────────────


class RoundPredictionsBundle(BaseModel):
    """Le previsioni di una giornata: partite + totali gol per lega."""

    match_predictions: list[MatchPredictionOut]
    round_predictions: list[RoundPredictionOut]


class RoundScoreOut(BaseModel):
    id: uuid.UUID
    player_id: uuid.UUID
    round_id: uuid.UUID
    sign_points: int
    exact_points: int
    total_goals_points: int
    weekend_bonus: int
    total_round_points: int

    model_config = {"from_attributes": True}
