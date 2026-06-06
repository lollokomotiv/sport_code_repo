"""Unit test della status machine del Round (grafo delle transizioni, puro)."""

from app.models.round import RoundStatus
from app.services.round import ALLOWED_TRANSITIONS, PLAYER_VISIBLE_STATUSES


def test_transition_graph_is_linear_forward_only():
    assert ALLOWED_TRANSITIONS[RoundStatus.draft] == {RoundStatus.open}
    assert ALLOWED_TRANSITIONS[RoundStatus.open] == {RoundStatus.closed}
    assert ALLOWED_TRANSITIONS[RoundStatus.closed] == {RoundStatus.completed}
    assert ALLOWED_TRANSITIONS[RoundStatus.completed] == set()


def test_no_backward_transitions():
    # completed è terminale; nessuno stato torna a 'draft'
    for targets in ALLOWED_TRANSITIONS.values():
        assert RoundStatus.draft not in targets


def test_player_visibility():
    assert RoundStatus.open in PLAYER_VISIBLE_STATUSES
    assert RoundStatus.completed in PLAYER_VISIBLE_STATUSES
    assert RoundStatus.draft not in PLAYER_VISIBLE_STATUSES
    assert RoundStatus.closed not in PLAYER_VISIBLE_STATUSES
