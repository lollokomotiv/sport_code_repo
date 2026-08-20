"""Unit test della status machine del Round (grafo delle transizioni, puro)."""

from app.models.round import RoundStatus
from app.services.round import ALLOWED_TRANSITIONS, PLAYER_VISIBLE_STATUSES


def test_transition_graph():
    assert ALLOWED_TRANSITIONS[RoundStatus.draft] == {RoundStatus.open}
    assert ALLOWED_TRANSITIONS[RoundStatus.open] == {RoundStatus.closed}
    # closed può andare avanti (completed) o essere RIAPERTA (open, solo pre-deadline)
    assert ALLOWED_TRANSITIONS[RoundStatus.closed] == {RoundStatus.completed, RoundStatus.open}
    assert ALLOWED_TRANSITIONS[RoundStatus.completed] == set()


def test_no_transition_back_to_draft():
    # nessuno stato torna a 'draft'; completed è terminale
    for targets in ALLOWED_TRANSITIONS.values():
        assert RoundStatus.draft not in targets
    assert ALLOWED_TRANSITIONS[RoundStatus.completed] == set()


def test_player_visibility():
    # open, closed (sola lettura) e completed sono visibili; draft no
    assert RoundStatus.open in PLAYER_VISIBLE_STATUSES
    assert RoundStatus.closed in PLAYER_VISIBLE_STATUSES
    assert RoundStatus.completed in PLAYER_VISIBLE_STATUSES
    assert RoundStatus.draft not in PLAYER_VISIBLE_STATUSES
