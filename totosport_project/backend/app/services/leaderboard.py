"""
Classifiche e bonus di fine stagione (REGOLAMENTO §5, §6).

Aggrega i RoundScore (giornate) con i punti tabellone (TablePrediction) e i bonus
finali (PlayerSeasonProfile). Funzioni FastAPI-free.

Composizione della classifica generale per giocatore:
    sum(RoundScore.total_round_points)        # giornate (segni+esatti+totgol+bonus weekend)
  + TablePrediction.total_points              # tabellone, GIÀ al netto della penalità mercato
  + bonus fine stagione (0/10 × 3)
"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.player_season_profile import PlayerSeasonProfile
from app.models.round import Round
from app.models.round_score import RoundScore
from app.models.season import Season, SeasonStatus
from app.models.table_prediction import TablePrediction
from app.models.user import User
from app.schemas.leaderboard import (
    LeaderboardEntry,
    MetricLeaderboardEntry,
    RoundLeaderboardEntry,
    SeasonBonusResult,
)

SEASON_BONUS_POINTS = 10


# ─── Funzione pura: bonus fine stagione ──────────────────────────────────────


def assign_season_bonus(
    ranking: list[tuple[uuid.UUID, int]], bonus_points: int = SEASON_BONUS_POINTS
) -> dict[uuid.UUID, int]:
    """Il/i migliore/i ricevono il bonus pieno (parità: tutti i pari merito)."""
    if not ranking:
        return {}
    best = max(pts for _, pts in ranking)
    if best <= 0:
        return {}  # nessun punteggio positivo → nessun bonus
    return {pid: bonus_points for pid, pts in ranking if pts == best}


def _ranks(sorted_pairs: list[tuple[uuid.UUID, int]]) -> dict[uuid.UUID, int]:
    """Ranking standard con parità (1, 2, 2, 4, ...). Input ordinato per score desc."""
    ranks: dict[uuid.UUID, int] = {}
    prev_score: int | None = None
    prev_rank = 0
    for i, (pid, score) in enumerate(sorted_pairs, start=1):
        if score != prev_score:
            prev_rank = i
            prev_score = score
        ranks[pid] = prev_rank
    return ranks


# ─── Aggregazioni base ───────────────────────────────────────────────────────


async def _round_aggregates(season_id: uuid.UUID, db: AsyncSession) -> dict[uuid.UUID, dict]:
    stmt = (
        select(
            RoundScore.player_id,
            func.sum(RoundScore.sign_points),
            func.sum(RoundScore.exact_points),
            func.sum(RoundScore.total_goals_points),
            func.sum(RoundScore.weekend_bonus),
            func.sum(RoundScore.total_round_points),
        )
        .join(Round, Round.id == RoundScore.round_id)
        .where(Round.season_id == season_id)
        .group_by(RoundScore.player_id)
    )
    result = await db.execute(stmt)
    out: dict[uuid.UUID, dict] = {}
    for pid, sign, exact, tg, wb, total in result.all():
        out[pid] = {
            "sign": sign or 0,
            "exact": exact or 0,
            "total_goals": tg or 0,
            "weekend": wb or 0,
            "total": total or 0,
        }
    return out


async def _tabellone_points(season_id: uuid.UUID, db: AsyncSession) -> dict[uuid.UUID, int]:
    stmt = select(TablePrediction.player_id, TablePrediction.total_points).where(
        TablePrediction.season_id == season_id,
        TablePrediction.total_points.is_not(None),
    )
    result = await db.execute(stmt)
    return {pid: pts for pid, pts in result.all()}


async def _profiles(season_id: uuid.UUID, db: AsyncSession) -> dict[uuid.UUID, PlayerSeasonProfile]:
    result = await db.execute(
        select(PlayerSeasonProfile).where(PlayerSeasonProfile.season_id == season_id)
    )
    return {p.player_id: p for p in result.scalars().all()}


async def _usernames(player_ids: set[uuid.UUID], db: AsyncSession) -> dict[uuid.UUID, str]:
    if not player_ids:
        return {}
    result = await db.execute(select(User.id, User.username).where(User.id.in_(player_ids)))
    return {pid: name for pid, name in result.all()}


# ─── Classifica generale ─────────────────────────────────────────────────────


async def general_leaderboard(season_id: uuid.UUID, db: AsyncSession) -> list[LeaderboardEntry]:
    rounds = await _round_aggregates(season_id, db)
    tabellone = await _tabellone_points(season_id, db)
    profiles = await _profiles(season_id, db)

    players = set(rounds) | set(tabellone) | set(profiles)
    usernames = await _usernames(players, db)

    rows = []
    for pid in players:
        r = rounds.get(pid, {})
        prof = profiles.get(pid)
        season_bonus = (
            (prof.season_bonus_signs + prof.season_bonus_exacts + prof.season_bonus_tabellone)
            if prof
            else 0
        )
        tab = tabellone.get(pid, 0)
        total = r.get("total", 0) + tab + season_bonus
        rows.append((pid, total, r, tab, season_bonus))

    ranks = _ranks(sorted(((pid, total) for pid, total, *_ in rows), key=lambda x: x[1], reverse=True))

    entries = [
        LeaderboardEntry(
            rank=ranks[pid],
            player_id=pid,
            username=usernames.get(pid, "?"),
            total_points=total,
            sign_points=r.get("sign", 0),
            exact_points=r.get("exact", 0),
            total_goals_points=r.get("total_goals", 0),
            weekend_bonus_total=r.get("weekend", 0),
            tabellone_points=tab,
            season_bonus_total=season_bonus,
        )
        for pid, total, r, tab, season_bonus in rows
    ]
    entries.sort(key=lambda e: (e.rank, e.username))
    return entries


# ─── Classifica di una giornata ──────────────────────────────────────────────


async def round_leaderboard(round_id: uuid.UUID, db: AsyncSession) -> list[RoundLeaderboardEntry]:
    result = await db.execute(select(RoundScore).where(RoundScore.round_id == round_id))
    scores = list(result.scalars().all())
    usernames = await _usernames({s.player_id for s in scores}, db)
    ranks = _ranks(
        sorted(((s.player_id, s.total_round_points) for s in scores), key=lambda x: x[1], reverse=True)
    )
    entries = [
        RoundLeaderboardEntry(
            rank=ranks[s.player_id],
            player_id=s.player_id,
            username=usernames.get(s.player_id, "?"),
            round_points=s.total_round_points,
            sign_points=s.sign_points,
            exact_points=s.exact_points,
            total_goals_points=s.total_goals_points,
            weekend_bonus=s.weekend_bonus,
        )
        for s in scores
    ]
    entries.sort(key=lambda e: (e.rank, e.username))
    return entries


# ─── Classifiche per singola metrica (bonus fine stagione) ────────────────────


async def _signs_scores(season_id: uuid.UUID, db: AsyncSession) -> dict[uuid.UUID, int]:
    return {pid: r["sign"] for pid, r in (await _round_aggregates(season_id, db)).items()}


async def _exacts_scores(season_id: uuid.UUID, db: AsyncSession) -> dict[uuid.UUID, int]:
    return {
        pid: r["exact"] + r["total_goals"]
        for pid, r in (await _round_aggregates(season_id, db)).items()
    }


async def _tabellone_scores(season_id: uuid.UUID, db: AsyncSession) -> dict[uuid.UUID, int]:
    return await _tabellone_points(season_id, db)


async def _metric_leaderboard(
    scores: dict[uuid.UUID, int], db: AsyncSession
) -> list[MetricLeaderboardEntry]:
    usernames = await _usernames(set(scores), db)
    ranks = _ranks(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    entries = [
        MetricLeaderboardEntry(
            rank=ranks[pid], player_id=pid, username=usernames.get(pid, "?"), points=pts
        )
        for pid, pts in scores.items()
    ]
    entries.sort(key=lambda e: (e.rank, e.username))
    return entries


async def signs_leaderboard(season_id, db) -> list[MetricLeaderboardEntry]:
    return await _metric_leaderboard(await _signs_scores(season_id, db), db)


async def exacts_leaderboard(season_id, db) -> list[MetricLeaderboardEntry]:
    return await _metric_leaderboard(await _exacts_scores(season_id, db), db)


async def tabellone_leaderboard(season_id, db) -> list[MetricLeaderboardEntry]:
    return await _metric_leaderboard(await _tabellone_scores(season_id, db), db)


# ─── Finalizzazione: bonus + chiusura stagione ───────────────────────────────


async def finalize_season(season: Season, db: AsyncSession) -> SeasonBonusResult:
    signs = await _signs_scores(season.id, db)
    exacts = await _exacts_scores(season.id, db)
    tabellone = await _tabellone_scores(season.id, db)

    bonus_signs = assign_season_bonus(list(signs.items()))
    bonus_exacts = assign_season_bonus(list(exacts.items()))
    bonus_tabellone = assign_season_bonus(list(tabellone.items()))

    winners = set(bonus_signs) | set(bonus_exacts) | set(bonus_tabellone)
    existing = await _profiles(season.id, db)
    for pid in winners:
        prof = existing.get(pid)
        if prof is None:
            prof = PlayerSeasonProfile(player_id=pid, season_id=season.id)
            db.add(prof)
        prof.season_bonus_signs = bonus_signs.get(pid, 0)
        prof.season_bonus_exacts = bonus_exacts.get(pid, 0)
        prof.season_bonus_tabellone = bonus_tabellone.get(pid, 0)

    season.status = SeasonStatus.closed
    await db.flush()

    return SeasonBonusResult(
        bonus_signs=list(bonus_signs.keys()),
        bonus_exacts=list(bonus_exacts.keys()),
        bonus_tabellone=list(bonus_tabellone.keys()),
    )
