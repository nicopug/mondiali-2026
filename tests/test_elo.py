"""Test del sistema Elo custom."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.features.elo import DEFAULT_ELO, EloSystem, classify_tournament


def test_elo_system_initializes_teams_at_default() -> None:
    """Team mai visto restituisce DEFAULT_ELO (1500)."""
    elo = EloSystem()
    assert elo.get("France") == DEFAULT_ELO
    assert elo.get("San Marino") == DEFAULT_ELO
    assert DEFAULT_ELO == 1500
