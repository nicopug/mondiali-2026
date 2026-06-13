"""Test per il parsing/validazione del tabellone knockout WC2026."""
from __future__ import annotations

import pytest

from mondiali.inference.bracket import (
    OFFICIAL_R32,
    ROUND_LABELS,
    BracketError,
    load_bracket,
)


def _filled(teams: list[str]) -> dict:
    """Costruisce un JSON-bracket valido a partire da 32 nomi (leaf order)."""
    pairs = [
        {"team_a": teams[2 * i], "team_b": teams[2 * i + 1]} for i in range(16)
    ]
    return {"bracket_r32": pairs}


def test_official_structure_has_16_matches_and_32_slots():
    assert len(OFFICIAL_R32) == 16
    slots = [s["slot_a"] for s in OFFICIAL_R32] + [s["slot_b"] for s in OFFICIAL_R32]
    assert len(slots) == 32
    # 12 winner-slots + 12 runner-up-slots + 8 third-slots
    assert sum(s.startswith("W-") for s in slots) == 12
    assert sum(s.startswith("RU-") for s in slots) == 12
    assert sum(s.startswith("3-") for s in slots) == 8


def test_round_labels_are_five_for_32_teams():
    assert ROUND_LABELS == ["R16", "QF", "SF", "Final", "Winner"]


def test_load_valid_bracket():
    teams = [f"T{i}" for i in range(32)]
    bracket = load_bracket(_filled(teams))
    assert len(bracket) == 16
    assert bracket[0] == {"team_a": "T0", "team_b": "T1"}
    assert bracket[15] == {"team_a": "T30", "team_b": "T31"}


def test_rejects_wrong_match_count():
    data = {"bracket_r32": [{"team_a": "A", "team_b": "B"}]}
    with pytest.raises(BracketError, match="16"):
        load_bracket(data)


def test_rejects_empty_team_name():
    teams = [f"T{i}" for i in range(32)]
    teams[5] = ""  # slot non compilato
    with pytest.raises(BracketError, match="vuoto|empty|compila"):
        load_bracket(_filled(teams))


def test_rejects_duplicate_team():
    teams = [f"T{i}" for i in range(32)]
    teams[7] = teams[0]  # squadra duplicata
    with pytest.raises(BracketError, match="dupli|distinct|32"):
        load_bracket(_filled(teams))
