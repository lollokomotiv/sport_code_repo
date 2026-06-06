"""Unit test dello scoring del tabellone — funzione pura, nessun DB."""

from app.models.season_outcome import SeasonOutcome
from app.models.table_prediction import TablePrediction
from app.services.tabellone import score_prediction


def pred(**kw) -> TablePrediction:
    return TablePrediction(**kw)


def outcome(**kw) -> SeasonOutcome:
    return SeasonOutcome(**kw)


# ─── Scudetto ──────────────────────────────────────────────────────────────────


def test_scudetto_team_and_points():
    b = score_prediction(
        pred(scudetto_team="Inter", scudetto_points_guess=90),
        outcome(scudetto_team="Inter", scudetto_points=90),
    )
    assert b["scudetto"] == 33  # 25 + 8


def test_scudetto_team_right_points_wrong():
    b = score_prediction(
        pred(scudetto_team="Inter", scudetto_points_guess=88),
        outcome(scudetto_team="Inter", scudetto_points=90),
    )
    assert b["scudetto"] == 25


def test_scudetto_wrong():
    b = score_prediction(
        pred(scudetto_team="Milan", scudetto_points_guess=90),
        outcome(scudetto_team="Inter", scudetto_points=90),
    )
    assert b["scudetto"] == 0


# ─── Retrocesse Serie A: matching a insiemi ─────────────────────────────────────


def test_relegated_a_set_based_ignores_slot():
    # Previste 3, ne retrocedono 2 ma in slot "diversi" → 50pt, lo slot non conta
    b = score_prediction(
        pred(relegated_a_1="Salernitana", relegated_a_2="Empoli", relegated_a_3="Frosinone"),
        outcome(relegated_a_1="Frosinone", relegated_a_2="Salernitana", relegated_a_3="Sassuolo"),
    )
    assert b["relegated_a_1"] + b["relegated_a_2"] + b["relegated_a_3"] == 50


def test_relegated_a_duplicate_not_double_counted():
    # Stessa squadra in due slot non vale doppio
    b = score_prediction(
        pred(relegated_a_1="Salernitana", relegated_a_2="Salernitana", relegated_a_3="Empoli"),
        outcome(relegated_a_1="Salernitana", relegated_a_2="Sassuolo", relegated_a_3="Cagliari"),
    )
    assert b["relegated_a_1"] + b["relegated_a_2"] + b["relegated_a_3"] == 25


# ─── Capocannoniere: parità ─────────────────────────────────────────────────────


def test_top_scorer_solo_winner():
    b = score_prediction(
        pred(top_scorer_a="Lautaro", top_scorer_a_goals=24),
        outcome(top_scorer_a="Lautaro", top_scorer_a_goals=24),
    )
    assert b["top_scorer_a"] == 23  # 15 + 8


def test_top_scorer_tie_divides_base_only():
    b = score_prediction(
        pred(top_scorer_a="Lautaro", top_scorer_a_goals=20),
        outcome(top_scorer_a="Lautaro, Vlahovic", top_scorer_a_goals=20),
    )
    assert b["top_scorer_a"] == 7 + 8  # floor(15/2)=7, +8 bonus gol


def test_top_scorer_tie_no_goal_bonus():
    b = score_prediction(
        pred(top_scorer_a="Lautaro", top_scorer_a_goals=99),
        outcome(top_scorer_a="Lautaro, Vlahovic", top_scorer_a_goals=20),
    )
    assert b["top_scorer_a"] == 7


# ─── Promozioni B: metodo sbagliato ─────────────────────────────────────────────


def test_promoted_direct_correct_method():
    b = score_prediction(
        pred(promoted_b_direct_1="Parma", promoted_b_direct_2="Como"),
        outcome(promoted_b_direct_1="Parma", promoted_b_direct_2="Como", promoted_b_playoff="Venezia"),
    )
    assert b["promoted_b_direct_2"] == 30


def test_promoted_direct_wrong_method_is_half():
    # Prevista come diretta ma promossa via playoff → 15
    b = score_prediction(
        pred(promoted_b_direct_2="Venezia"),
        outcome(promoted_b_direct_1="Parma", promoted_b_direct_2="Como", promoted_b_playoff="Venezia"),
    )
    assert b["promoted_b_direct_2"] == 15


def test_promoted_first_points_bonus():
    b = score_prediction(
        pred(promoted_b_direct_1="Parma", promoted_b_first_points=78),
        outcome(promoted_b_direct_1="Parma", promoted_b_first_points=78, promoted_b_playoff="Venezia"),
    )
    assert b["promoted_b_direct_1"] == 38  # 30 + 8


def test_promoted_playoff_wrong_method_is_half():
    # Prevista come "promossa col playoff" ma promossa direttamente → 15
    b = score_prediction(
        pred(promoted_b_playoff="Parma"),
        outcome(
            promoted_b_direct_1="Parma", promoted_b_direct_2="Como",
            promoted_b_playoff="Venezia", playoffs_held=True,
        ),
    )
    assert b["promoted_b_playoff"] == 15


# ─── Playoff a insiemi ─────────────────────────────────────────────────────────


def test_playoff_teams_set_based():
    b = score_prediction(
        pred(playoff_b_1="Brescia", playoff_b_2="Catanzaro", playoff_b_3="Cremonese"),
        outcome(
            playoffs_held=True,
            playoff_b_1="Cremonese", playoff_b_2="Brescia", playoff_b_3="Palermo",
            playoff_b_4="Sampdoria", playoff_b_5="Catanzaro", playoff_b_6="Pisa",
        ),
    )
    total = sum(b[f] for f in ("playoff_b_1", "playoff_b_2", "playoff_b_3", "playoff_b_4", "playoff_b_5", "playoff_b_6"))
    assert total == 60  # 3 azzeccate × 20


# ─── Alternative "no playoff / no playout" ──────────────────────────────────────


def test_no_playoffs_awards_40_if_promoted_matches():
    b = score_prediction(
        pred(promoted_b_playoff="Sampdoria", playoff_b_1="Brescia"),
        outcome(playoffs_held=False, promoted_b_playoff="Sampdoria"),
    )
    assert b["promoted_b_playoff"] == 40
    assert b["playoff_b_1"] == 0  # le voci playoff sono azzerate


def test_no_playouts_awards_35_if_relegated_matches():
    b = score_prediction(
        pred(relegated_b_c_playout="Ternana", playout_b_1="Bari"),
        outcome(playout_held=False, relegated_b_c_playout="Ternana"),
    )
    assert b["relegated_b_c_playout"] == 35
    assert b["playout_b_1"] == 0


# ─── Coppe ──────────────────────────────────────────────────────────────────────


def test_coppe_winners():
    b = score_prediction(
        pred(
            coppa_italia_winner="Juventus", champions_winner="Real Madrid",
            europa_winner="Atalanta", conference_winner="Fiorentina",
        ),
        outcome(
            coppa_italia_winner="Juventus", champions_winner="Real Madrid",
            europa_winner="Atalanta", conference_winner="Roma",
        ),
    )
    assert b["coppa_italia_winner"] == 25
    assert b["champions_winner"] == 25
    assert b["europa_winner"] == 25
    assert b["conference_winner"] == 0
