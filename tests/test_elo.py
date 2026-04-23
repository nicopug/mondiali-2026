"""Test del sistema Elo custom."""
from __future__ import annotations

import pytest

from mondiali.features.elo import DEFAULT_ELO, EloSystem


def test_elo_system_initializes_teams_at_default() -> None:
    """Team mai visto restituisce DEFAULT_ELO (1500)."""
    elo = EloSystem()
    assert elo.get("France") == DEFAULT_ELO
    assert elo.get("San Marino") == DEFAULT_ELO
    assert DEFAULT_ELO == 1500


def test_elo_update_home_win_zero_sum() -> None:
    """Dopo una vittoria casa, la somma dei rating si conserva (zero-sum)."""
    elo = EloSystem()
    elo.update(home="France", away="Brazil", home_goals=2, away_goals=0, k_factor=30, neutral=False)
    total = elo.get("France") + elo.get("Brazil")
    assert total == pytest.approx(2 * DEFAULT_ELO, abs=0.01)


def test_elo_update_home_win_increases_home_rating() -> None:
    """Vittoria casa → rating casa aumenta, ospite diminuisce."""
    elo = EloSystem()
    elo.update(home="France", away="Brazil", home_goals=2, away_goals=0, k_factor=30, neutral=False)
    assert elo.get("France") > DEFAULT_ELO
    assert elo.get("Brazil") < DEFAULT_ELO


def test_elo_update_draw_between_equal_teams_neutral_no_change() -> None:
    """Pareggio tra squadre di pari Elo in venue neutral → nessun cambiamento."""
    elo = EloSystem()
    elo.update(home="A", away="B", home_goals=1, away_goals=1, k_factor=30, neutral=True)
    assert elo.get("A") == pytest.approx(DEFAULT_ELO, abs=0.01)
    assert elo.get("B") == pytest.approx(DEFAULT_ELO, abs=0.01)


def test_elo_update_away_win_increases_away_rating() -> None:
    """Vittoria trasferta → rating ospite aumenta."""
    elo = EloSystem()
    elo.update(home="A", away="B", home_goals=0, away_goals=3, k_factor=30, neutral=False)
    assert elo.get("B") > DEFAULT_ELO
    assert elo.get("A") < DEFAULT_ELO


def test_elo_update_magnitude_proportional_to_k() -> None:
    """K=60 produce delta doppio rispetto a K=30 a parità di condizioni."""
    elo_a = EloSystem()
    elo_b = EloSystem()
    elo_a.update(home="X", away="Y", home_goals=1, away_goals=0, k_factor=30, neutral=True)
    elo_b.update(home="X", away="Y", home_goals=1, away_goals=0, k_factor=60, neutral=True)
    delta_a = elo_a.get("X") - DEFAULT_ELO
    delta_b = elo_b.get("X") - DEFAULT_ELO
    assert delta_b == pytest.approx(2 * delta_a, abs=0.01)
