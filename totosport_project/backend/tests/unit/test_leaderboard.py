"""Unit test delle funzioni pure della leaderboard."""

import uuid

from app.services.leaderboard import _ranks, assign_season_bonus


def _ids(n):
    return [uuid.uuid4() for _ in range(n)]


def test_assign_bonus_single_winner():
    a, b, c = _ids(3)
    assert assign_season_bonus([(a, 30), (b, 20), (c, 10)]) == {a: 10}


def test_assign_bonus_tie():
    a, b, c = _ids(3)
    res = assign_season_bonus([(a, 30), (b, 30), (c, 10)])
    assert res == {a: 10, b: 10}


def test_assign_bonus_empty():
    assert assign_season_bonus([]) == {}


def test_assign_bonus_all_zero_no_bonus():
    a, b = _ids(2)
    # Nessun punteggio positivo → nessun bonus
    assert assign_season_bonus([(a, 0), (b, 0)]) == {}


def test_ranks_with_ties():
    a, b, c, d = _ids(4)
    ranks = _ranks([(a, 30), (b, 20), (c, 20), (d, 10)])
    assert ranks == {a: 1, b: 2, c: 2, d: 4}  # ranking standard: il 3° posto salta
