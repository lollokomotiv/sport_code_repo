"""
Logica di business delle previsioni settimanali (REGOLAMENTO §3).

Le previsioni si inseriscono/aggiornano (upsert) solo quando la giornata è 'open'
e la deadline non è passata. Funzioni FastAPI-free: sollevano eccezioni di dominio.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.match import Match
from app.models.match_prediction import MatchPrediction
from app.models.round import Competition, Round, RoundStatus
from app.models.round_prediction import RoundPrediction


# ─── Eccezioni di dominio ────────────────────────────────────────────────────


class PredictionError(Exception):
    pass


class PredictionsClosed(PredictionError):
    """Giornata non 'open' o deadline superata: niente inserimenti/modifiche."""


class MatchNotFound(PredictionError):
    pass


class RoundNotFound(PredictionError):
    pass


class CompetitionNotInRound(PredictionError):
    """Nessuna partita di quella lega nella giornata: niente totale gol."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_open(rnd: Round) -> None:
    if rnd.status != RoundStatus.open:
        raise PredictionsClosed("Le previsioni sono ammesse solo a giornata aperta")
    if rnd.deadline is not None and _now() > rnd.deadline:
        raise PredictionsClosed("Deadline della giornata superata")


# ─── Upsert previsioni ───────────────────────────────────────────────────────


async def submit_match_prediction(
    player_id: uuid.UUID, data: dict, db: AsyncSession
) -> MatchPrediction:
    match = await db.get(Match, data["match_id"])
    if match is None:
        raise MatchNotFound("Partita non trovata")
    rnd = await db.get(Round, match.round_id)
    _ensure_open(rnd)

    result = await db.execute(
        select(MatchPrediction).where(
            MatchPrediction.player_id == player_id,
            MatchPrediction.match_id == match.id,
        )
    )
    pred = result.scalar_one_or_none()
    if pred is None:
        pred = MatchPrediction(player_id=player_id, match_id=match.id)
        db.add(pred)
    pred.predicted_sign = data["predicted_sign"]
    # Il risultato esatto conta solo sulle partite che lo richiedono; altrove lo ignoriamo
    if match.requires_exact_score:
        pred.predicted_home_goals = data.get("predicted_home_goals")
        pred.predicted_away_goals = data.get("predicted_away_goals")
    else:
        pred.predicted_home_goals = None
        pred.predicted_away_goals = None
    pred.submitted_at = _now()
    await db.flush()
    return pred


async def submit_round_prediction(
    player_id: uuid.UUID, data: dict, db: AsyncSession
) -> RoundPrediction:
    rnd = await db.get(Round, data["round_id"])
    if rnd is None:
        raise RoundNotFound("Giornata non trovata")
    _ensure_open(rnd)

    competition: Competition = data["competition"]
    count = await db.scalar(
        select(func.count())
        .select_from(Match)
        .where(Match.round_id == rnd.id, Match.competition == competition)
    )
    if not count:
        raise CompetitionNotInRound(
            f"Nessuna partita di {competition.value} in questa giornata"
        )

    result = await db.execute(
        select(RoundPrediction).where(
            RoundPrediction.player_id == player_id,
            RoundPrediction.round_id == rnd.id,
            RoundPrediction.competition == competition,
        )
    )
    pred = result.scalar_one_or_none()
    if pred is None:
        pred = RoundPrediction(player_id=player_id, round_id=rnd.id, competition=competition)
        db.add(pred)
    pred.total_goals_guess = data["total_goals_guess"]
    pred.submitted_at = _now()
    await db.flush()
    return pred


# ─── Query ───────────────────────────────────────────────────────────────────


async def get_my_match_predictions(
    player_id: uuid.UUID, round_id: uuid.UUID, db: AsyncSession
) -> list[MatchPrediction]:
    result = await db.execute(
        select(MatchPrediction)
        .join(Match, Match.id == MatchPrediction.match_id)
        .where(Match.round_id == round_id, MatchPrediction.player_id == player_id)
    )
    return list(result.scalars().all())


async def get_my_round_predictions(
    player_id: uuid.UUID, round_id: uuid.UUID, db: AsyncSession
) -> list[RoundPrediction]:
    result = await db.execute(
        select(RoundPrediction).where(
            RoundPrediction.round_id == round_id,
            RoundPrediction.player_id == player_id,
        )
    )
    return list(result.scalars().all())


async def get_match_prediction_history(
    player_id: uuid.UUID, db: AsyncSession, limit: int = 100, offset: int = 0
) -> list[MatchPrediction]:
    result = await db.execute(
        select(MatchPrediction)
        .where(MatchPrediction.player_id == player_id)
        .order_by(MatchPrediction.submitted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def list_round_match_predictions(
    round_id: uuid.UUID, db: AsyncSession
) -> list[MatchPrediction]:
    result = await db.execute(
        select(MatchPrediction)
        .join(Match, Match.id == MatchPrediction.match_id)
        .where(Match.round_id == round_id)
    )
    return list(result.scalars().all())


async def list_match_predictions(
    match_id: uuid.UUID, db: AsyncSession
) -> list[MatchPrediction]:
    result = await db.execute(
        select(MatchPrediction).where(MatchPrediction.match_id == match_id)
    )
    return list(result.scalars().all())


async def list_round_total_goals_predictions(
    round_id: uuid.UUID, db: AsyncSession
) -> list[RoundPrediction]:
    result = await db.execute(
        select(RoundPrediction).where(RoundPrediction.round_id == round_id)
    )
    return list(result.scalars().all())
