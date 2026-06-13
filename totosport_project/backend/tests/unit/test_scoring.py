"""Unit test del motore di scoring giornata — funzioni pure, nessun DB."""

from app.services.scoring import (
    assign_weekend_bonus,
    derive_sign,
    score_match,
    score_total_goals,
)


# ─── derive_sign ────────────────────────────────────────────────────────────────


def test_derive_sign():
    assert derive_sign(2, 1) == "1"
    assert derive_sign(1, 1) == "X"
    assert derive_sign(0, 2) == "2"


# ─── score_match: partite SOLO-SEGNO (requires_exact=False) ─────────────────────


def test_sign_only_correct():
    # segno scelto giusto, nessun risultato esatto richiesto
    assert score_match("1", None, None, 2, 1, False) == (1, 0)


def test_sign_only_wrong():
    assert score_match("2", None, None, 2, 1, False) == (0, 0)


def test_exact_goals_ignored_when_not_required():
    # Anche se i goal coincidono, senza requires_exact niente bonus (solo il segno)
    assert score_match("1", 2, 1, 2, 1, False) == (1, 0)


# ─── score_match: partite con RISULTATO ESATTO (requires_exact=True) ─────────────


def test_score_match_home_exact():
    assert score_match("1", 2, 1, 2, 1, True) == (1, 5)


def test_score_match_draw_exact():
    assert score_match("X", 1, 1, 1, 1, True) == (1, 7)


def test_score_match_away_exact():
    assert score_match("2", 0, 2, 0, 2, True) == (1, 9)


def test_score_match_high_scoring_draw():
    # 3-3: pareggio esatto con 6 gol → 7 + 2
    assert score_match("X", 3, 3, 3, 3, True) == (1, 9)


def test_score_match_high_scoring_home():
    # 3-2 esatto (5 gol) → 5 + 2
    assert score_match("1", 3, 2, 3, 2, True) == (1, 7)


def test_score_match_sign_right_exact_wrong():
    # segno giusto (casa), risultato sbagliato
    assert score_match("1", 2, 1, 1, 0, True) == (1, 0)


def test_score_match_wrong_sign():
    assert score_match("2", 0, 1, 1, 0, True) == (0, 0)


# ─── score_total_goals ──────────────────────────────────────────────────────────


def test_total_goals_correct():
    assert score_total_goals(5, [(2, 1), (1, 1)]) == 3


def test_total_goals_wrong():
    assert score_total_goals(6, [(2, 1), (1, 1)]) == 0


# ─── assign_weekend_bonus ───────────────────────────────────────────────────────


def test_weekend_bonus_no_ties():
    res = assign_weekend_bonus([("A", 10), ("B", 8), ("C", 6), ("D", 4)])
    assert res == {"A": 6, "B": 4, "C": 2}


def test_weekend_bonus_tie_gets_higher_position():
    # B e C pari merito per il 2°: prendono entrambi 4 (bonus del posto più alto),
    # il 3° posto non viene assegnato (decisione admin / REGOLAMENTO §5).
    res = assign_weekend_bonus([("A", 10), ("B", 8), ("C", 8), ("D", 5)])
    assert res == {"A": 6, "B": 4, "C": 4}


def test_weekend_bonus_tie_for_first():
    res = assign_weekend_bonus([("A", 9), ("B", 9), ("C", 5)])
    # entrambi 1° → 6 ciascuno; poi rank passa a 3 → C prende il 3° (2)
    assert res == {"A": 6, "B": 6, "C": 2}
