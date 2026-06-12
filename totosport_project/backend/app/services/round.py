"""
Logica di business di giornate e partite (Fase 4).

Funzioni pure rispetto a FastAPI: sollevano eccezioni di DOMINIO che i router
traducono in HTTP. La status machine del Round è:

    draft ──► open ──► closed ──► completed   (nessun ritorno indietro)
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.match import Match
from app.models.round import Round, RoundStatus
from app.models.staged_fixture import StagedFixture
from app.services.scoring import finalize_round, score_match_and_persist

# Transizioni di stato ammesse
ALLOWED_TRANSITIONS: dict[RoundStatus, set[RoundStatus]] = {
    RoundStatus.draft: {RoundStatus.open},
    RoundStatus.open: {RoundStatus.closed},
    RoundStatus.closed: {RoundStatus.completed},
    RoundStatus.completed: set(),
}

# Stati visibili ai giocatori (GET /rounds)
PLAYER_VISIBLE_STATUSES = (RoundStatus.open, RoundStatus.completed)


# ─── Eccezioni di dominio ────────────────────────────────────────────────────


class RoundError(Exception):
    """Base per gli errori di dominio di round/match."""


class RoundNotFound(RoundError):
    pass


class MatchNotFound(RoundError):
    pass


class StagedFixtureNotFound(RoundError):
    pass


class InvalidStatusTransition(RoundError):
    pass


class RoundStateViolation(RoundError):
    """Operazione non ammessa nello stato corrente del round."""


class RoundPreconditionError(RoundError):
    """Precondizione di transizione non soddisfatta."""


class DuplicateMatch(RoundError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Query ───────────────────────────────────────────────────────────────────


async def get_round(round_id: uuid.UUID, db: AsyncSession) -> Round | None:
    result = await db.execute(select(Round).where(Round.id == round_id))
    return result.scalar_one_or_none()


async def list_rounds(
    season_id: uuid.UUID,
    db: AsyncSession,
    statuses: tuple[RoundStatus, ...] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Round]:
    query = select(Round).where(Round.season_id == season_id)
    if statuses is not None:
        query = query.where(Round.status.in_(statuses))
    query = query.order_by(Round.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    return list(result.scalars().all())


async def list_matches(round_id: uuid.UUID, db: AsyncSession) -> list[Match]:
    result = await db.execute(
        select(Match).where(Match.round_id == round_id).order_by(Match.kickoff)
    )
    return list(result.scalars().all())


async def get_match(match_id: uuid.UUID, db: AsyncSession) -> Match | None:
    result = await db.execute(select(Match).where(Match.id == match_id))
    return result.scalar_one_or_none()


# ─── Round: CRUD + status machine ─────────────────────────────────────────────


async def create_round(season_id: uuid.UUID, data: dict, db: AsyncSession) -> Round:
    rnd = Round(season_id=season_id, **data)
    db.add(rnd)
    await db.flush()
    return rnd


async def delete_round(rnd: Round, db: AsyncSession) -> None:
    if rnd.status != RoundStatus.draft:
        raise RoundStateViolation("Si possono eliminare solo le giornate in stato 'draft'")
    await db.delete(rnd)
    await db.flush()


async def transition_status(rnd: Round, new_status: RoundStatus, db: AsyncSession) -> Round:
    if new_status not in ALLOWED_TRANSITIONS[rnd.status]:
        raise InvalidStatusTransition(
            f"Transizione non ammessa: {rnd.status.value} → {new_status.value}"
        )

    if rnd.status == RoundStatus.draft and new_status == RoundStatus.open:
        count = await db.scalar(
            select(func.count()).select_from(Match).where(Match.round_id == rnd.id)
        )
        if not count:
            raise RoundPreconditionError("La giornata deve avere almeno una partita per essere aperta")
        if rnd.deadline is None or rnd.deadline <= _now():
            raise RoundPreconditionError("La deadline deve essere impostata e nel futuro")

    finalizing = rnd.status == RoundStatus.closed and new_status == RoundStatus.completed
    if finalizing:
        matches = await list_matches(rnd.id, db)
        missing = [m for m in matches if m.actual_home_goals is None or m.actual_away_goals is None]
        if missing:
            raise RoundPreconditionError("Tutti i risultati devono essere inseriti prima di completare")

    rnd.status = new_status
    await db.flush()

    # Alla chiusura definitiva: calcola RoundScore aggregato + bonus weekend (Fase 5)
    if finalizing:
        await finalize_round(rnd.id, db)

    return rnd


# ─── Match: aggiunta / rimozione / risultato ──────────────────────────────────


async def add_match(rnd: Round, data: dict, db: AsyncSession) -> Match:
    if rnd.status != RoundStatus.draft:
        raise RoundStateViolation("Le partite si aggiungono solo a giornate in stato 'draft'")

    staged_fixture_id = data.get("staged_fixture_id")
    if staged_fixture_id is not None:
        result = await db.execute(
            select(StagedFixture).where(StagedFixture.id == staged_fixture_id)
        )
        fixture = result.scalar_one_or_none()
        if fixture is None:
            raise StagedFixtureNotFound("Staged fixture non trovata")
        home_team, away_team = fixture.home_team, fixture.away_team
        kickoff = fixture.kickoff
        api_fixture_id = fixture.api_fixture_id
        competition = fixture.competition
        fixture.added_to_round = True
    else:
        home_team = data.get("home_team")
        away_team = data.get("away_team")
        kickoff = data.get("kickoff")
        competition = data.get("competition")
        api_fixture_id = None
        if not home_team or not away_team:
            raise RoundError("Servono home_team e away_team (o uno staged_fixture_id)")
        if competition is None:
            raise RoundError("La competizione della partita è obbligatoria")

    # Niente due partite con le stesse squadre nella stessa giornata
    dup = await db.scalar(
        select(func.count())
        .select_from(Match)
        .where(
            Match.round_id == rnd.id,
            Match.home_team == home_team,
            Match.away_team == away_team,
        )
    )
    if dup:
        raise DuplicateMatch(f"{home_team} - {away_team} è già presente nella giornata")

    match = Match(
        round_id=rnd.id,
        competition=competition,
        home_team=home_team,
        away_team=away_team,
        kickoff=kickoff,
        api_fixture_id=api_fixture_id,
    )
    db.add(match)
    await db.flush()
    return match


async def delete_match(match: Match, rnd: Round, db: AsyncSession) -> None:
    if rnd.status != RoundStatus.draft:
        raise RoundStateViolation("Le partite si rimuovono solo da giornate in stato 'draft'")
    await db.delete(match)
    await db.flush()


async def set_match_result(
    match: Match, rnd: Round, home_goals: int, away_goals: int, db: AsyncSession
) -> Match:
    if rnd.status != RoundStatus.closed:
        raise RoundStateViolation(
            "Il risultato si imposta solo quando la giornata è in stato 'closed'"
        )
    match.actual_home_goals = home_goals
    match.actual_away_goals = away_goals
    await db.flush()
    # Scoring per-partita (Fase 5): aggiorna points_earned di ogni previsione.
    # L'aggregato di giornata + bonus weekend avviene in finalize_round (closed→completed).
    await score_match_and_persist(match.id, db)
    return match
