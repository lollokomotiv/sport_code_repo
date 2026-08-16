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
from app.models.user import User, UserRole
from app.services.scoring import derive_sign


def _max_dt(a: datetime | None, b: datetime | None) -> datetime | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


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


class SignResultMismatch(PredictionError):
    """Il segno scelto è in contrasto col risultato esatto indicato."""


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
        home = data.get("predicted_home_goals")
        away = data.get("predicted_away_goals")
        # Coerenza: se è indicato il risultato esatto, il segno deve corrispondere.
        if home is not None and away is not None and derive_sign(home, away) != data["predicted_sign"]:
            raise SignResultMismatch(
                "Il segno scelto non corrisponde al risultato esatto indicato"
            )
        pred.predicted_home_goals = home
        pred.predicted_away_goals = away
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


# ─── F1: schedine di tutti i giocatori, raggruppate (con username) ─────────────


async def list_round_predictions_grouped(round_id: uuid.UUID, db: AsyncSession) -> list[dict]:
    """
    Tutte le schedine compilate della giornata, raggruppate per giocatore, con
    lo username. Solo i giocatori con almeno una previsione compaiono.
    Restituisce dict pronti per PlayerPredictions (gli ORM sono mappabili con
    from_attributes). L'AUTORIZZAZIONE (finestra chiusa) è a carico del router.
    """
    mres = await db.execute(
        select(MatchPrediction, User.username)
        .join(Match, Match.id == MatchPrediction.match_id)
        .join(User, User.id == MatchPrediction.player_id)
        .where(Match.round_id == round_id)
    )
    rres = await db.execute(
        select(RoundPrediction, User.username)
        .join(User, User.id == RoundPrediction.player_id)
        .where(RoundPrediction.round_id == round_id)
    )

    grouped: dict[uuid.UUID, dict] = {}

    def _ensure(pid: uuid.UUID, username: str) -> dict:
        if pid not in grouped:
            grouped[pid] = {
                "player_id": pid,
                "username": username,
                "submitted_at": None,
                "match_predictions": [],
                "round_predictions": [],
            }
        return grouped[pid]

    for mp, username in mres.all():
        g = _ensure(mp.player_id, username)
        g["match_predictions"].append(mp)
        g["submitted_at"] = _max_dt(g["submitted_at"], mp.submitted_at)
    for rp, username in rres.all():
        g = _ensure(rp.player_id, username)
        g["round_predictions"].append(rp)
        g["submitted_at"] = _max_dt(g["submitted_at"], rp.submitted_at)

    return sorted(grouped.values(), key=lambda g: g["username"].lower())


# ─── F2: stato compilazione (solo conteggi, nessun contenuto) ──────────────────


async def round_submission_status(round_id: uuid.UUID, db: AsyncSession) -> dict:
    """
    Per ogni giocatore ATTIVO: quante partite ha pronosticato su quante totali,
    quanti totali-gol ha inviato, e l'ultimo submitted_at. Nessun contenuto.
    """
    total_matches = (
        await db.scalar(
            select(func.count()).select_from(Match).where(Match.round_id == round_id)
        )
        or 0
    )

    mp_rows = await db.execute(
        select(
            MatchPrediction.player_id,
            func.count(),
            func.max(MatchPrediction.submitted_at),
        )
        .join(Match, Match.id == MatchPrediction.match_id)
        .where(Match.round_id == round_id)
        .group_by(MatchPrediction.player_id)
    )
    mp_map = {pid: (cnt, last) for pid, cnt, last in mp_rows.all()}

    rp_rows = await db.execute(
        select(
            RoundPrediction.player_id,
            func.count(),
            func.max(RoundPrediction.submitted_at),
        )
        .where(RoundPrediction.round_id == round_id)
        .group_by(RoundPrediction.player_id)
    )
    rp_map = {pid: (cnt, last) for pid, cnt, last in rp_rows.all()}

    players_res = await db.execute(
        select(User)
        .where(User.role == UserRole.player, User.is_active.is_(True))
        .order_by(User.username)
    )

    out = []
    for u in players_res.scalars().all():
        m_cnt, m_last = mp_map.get(u.id, (0, None))
        r_cnt, r_last = rp_map.get(u.id, (0, None))
        out.append(
            {
                "player_id": u.id,
                "username": u.username,
                "matches_predicted": m_cnt,
                "total_matches": total_matches,
                "round_goals_count": r_cnt,
                "submitted_at": _max_dt(m_last, r_last),
            }
        )
    return {"total_matches": total_matches, "players": out}
