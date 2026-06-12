"""
Motore di scoring delle giornate (REGOLAMENTO §3, §4, §5; CLAUDE.md §8).

Le funzioni "pure" non dipendono da FastAPI né dal DB e sono testabili da sole.
Le funzioni "persistenti" applicano i risultati al DB:
- `score_match_and_persist`: punti per-partita, dopo l'inserimento di un risultato.
- `finalize_round`: aggrega il RoundScore di ogni giocatore e assegna il bonus
  weekend, quando l'admin porta la giornata a 'completed'.
"""

import uuid
from collections import defaultdict
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.match import Match
from app.models.match_prediction import MatchPrediction
from app.models.round_prediction import RoundPrediction
from app.models.round_score import RoundScore

# ─── Costanti (REGOLAMENTO §4-5) — non modificare senza aggiornare il REGOLAMENTO ─
SIGN_POINT = 1
EXACT_BONUS = {"1": 5, "X": 7, "2": 9}
HIGH_SCORING_BONUS = 2
HIGH_SCORING_THRESHOLD = 5
TOTAL_GOALS_POINTS = 3
WEEKEND_BONUS = {1: 6, 2: 4, 3: 2}


# ─── Funzioni pure ────────────────────────────────────────────────────────────


def derive_sign(home: int, away: int) -> Literal["1", "X", "2"]:
    if home > away:
        return "1"
    if home == away:
        return "X"
    return "2"


def score_match(
    pred_home: int, pred_away: int, actual_home: int, actual_away: int
) -> tuple[int, int]:
    """
    Restituisce (sign_points, exact_points).
    - sign_points: 1 se il segno è giusto, 0 altrimenti.
    - exact_points: 5/7/9 (+2 se 5+ gol totali) se il risultato esatto è giusto, 0 altrimenti.
    Totale partita = sign_points + exact_points.
    """
    sign_pts = SIGN_POINT if derive_sign(pred_home, pred_away) == derive_sign(actual_home, actual_away) else 0
    exact_pts = 0
    if pred_home == actual_home and pred_away == actual_away:
        exact_pts = EXACT_BONUS[derive_sign(actual_home, actual_away)]
        if actual_home + actual_away >= HIGH_SCORING_THRESHOLD:
            exact_pts += HIGH_SCORING_BONUS
    return sign_pts, exact_pts


def score_total_goals(guess: int, matches: list[tuple[int, int]]) -> int:
    """matches: lista di (gol_casa, gol_trasferta) delle partite con risultato noto."""
    actual_total = sum(h + a for h, a in matches)
    return TOTAL_GOALS_POINTS if guess == actual_total else 0


def assign_weekend_bonus(scores: list[tuple[uuid.UUID, int]]) -> dict[uuid.UUID, int]:
    """
    Restituisce {player_id: bonus}. In caso di parità, il bonus della posizione è
    assegnato INTEGRALMENTE a tutti i pari merito (REGOLAMENTO §5).
    """
    ordered = sorted(scores, key=lambda x: x[1], reverse=True)
    result: dict[uuid.UUID, int] = {}
    rank = 1
    i = 0
    while i < len(ordered) and rank <= 3:
        group_score = ordered[i][1]
        group = [pid for pid, pts in ordered if pts == group_score]
        bonus = WEEKEND_BONUS.get(rank, 0)
        for pid in group:
            result[pid] = bonus
        rank += len(group)
        i += len(group)
    return result


# ─── Funzioni persistenti ─────────────────────────────────────────────────────


async def score_match_and_persist(match_id: uuid.UUID, db: AsyncSession) -> None:
    """Aggiorna `points_earned` di ogni previsione della partita (no aggregato)."""
    match = await db.get(Match, match_id)
    if match is None or match.actual_home_goals is None or match.actual_away_goals is None:
        return
    result = await db.execute(
        select(MatchPrediction).where(MatchPrediction.match_id == match_id)
    )
    for pred in result.scalars().all():
        sign_pts, exact_pts = score_match(
            pred.predicted_home_goals,
            pred.predicted_away_goals,
            match.actual_home_goals,
            match.actual_away_goals,
        )
        pred.points_earned = sign_pts + exact_pts
    await db.flush()


async def finalize_round(round_id: uuid.UUID, db: AsyncSession) -> list[RoundScore]:
    """
    Calcola il RoundScore di ogni giocatore della giornata e assegna il bonus weekend.
    Da chiamare alla transizione 'closed' → 'completed' (tutti i risultati presenti).
    """
    matches_res = await db.execute(select(Match).where(Match.round_id == round_id))
    matches = list(matches_res.scalars().all())
    matches_by_id = {m.id: m for m in matches}
    match_ids = list(matches_by_id.keys())

    # Gol reali raggruppati per lega (per il totale gol)
    actual_by_comp: dict = defaultdict(list)
    for m in matches:
        if m.actual_home_goals is not None and m.actual_away_goals is not None:
            actual_by_comp[m.competition].append((m.actual_home_goals, m.actual_away_goals))

    sign_by_player: dict[uuid.UUID, int] = defaultdict(int)
    exact_by_player: dict[uuid.UUID, int] = defaultdict(int)
    tg_by_player: dict[uuid.UUID, int] = defaultdict(int)
    players: set[uuid.UUID] = set()

    # Punti segno + esatto dalle previsioni partita
    if match_ids:
        mp_res = await db.execute(
            select(MatchPrediction).where(MatchPrediction.match_id.in_(match_ids))
        )
        for p in mp_res.scalars().all():
            m = matches_by_id[p.match_id]
            if m.actual_home_goals is None or m.actual_away_goals is None:
                continue
            s, e = score_match(
                p.predicted_home_goals, p.predicted_away_goals,
                m.actual_home_goals, m.actual_away_goals,
            )
            sign_by_player[p.player_id] += s
            exact_by_player[p.player_id] += e
            players.add(p.player_id)

    # Punti totale gol, per lega
    rp_res = await db.execute(
        select(RoundPrediction).where(RoundPrediction.round_id == round_id)
    )
    for rp in rp_res.scalars().all():
        pts = score_total_goals(rp.total_goals_guess, actual_by_comp.get(rp.competition, []))
        rp.points_earned = pts
        tg_by_player[rp.player_id] += pts
        players.add(rp.player_id)

    # Bonus weekend sul totale (segno + esatto + totale gol), NON incluso il bonus stesso
    base_totals = [
        (pid, sign_by_player[pid] + exact_by_player[pid] + tg_by_player[pid]) for pid in players
    ]
    bonus = assign_weekend_bonus(base_totals)

    # Upsert RoundScore per ogni giocatore
    existing_res = await db.execute(select(RoundScore).where(RoundScore.round_id == round_id))
    existing = {rs.player_id: rs for rs in existing_res.scalars().all()}

    scores: list[RoundScore] = []
    for pid in players:
        rs = existing.get(pid)
        if rs is None:
            rs = RoundScore(player_id=pid, round_id=round_id)
            db.add(rs)
        rs.sign_points = sign_by_player[pid]
        rs.exact_points = exact_by_player[pid]
        rs.total_goals_points = tg_by_player[pid]
        rs.weekend_bonus = bonus.get(pid, 0)
        rs.total_round_points = (
            rs.sign_points + rs.exact_points + rs.total_goals_points + rs.weekend_bonus
        )
        scores.append(rs)

    await db.flush()
    return scores
