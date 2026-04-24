"""Test del sistema Elo custom."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.config import CONFIG, K_FACTORS
from mondiali.data.ingestion import load_international_results
from mondiali.features.elo import DEFAULT_ELO, EloSystem, classify_tournament


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


@pytest.mark.parametrize(
    ("tournament", "expected"),
    [
        ("FIFA World Cup", "world_cup"),
        ("FIFA World Cup qualification", "qualification"),
        ("UEFA Euro", "continental"),
        ("UEFA Euro qualification", "qualification"),
        ("Copa América", "continental"),
        ("African Cup of Nations", "continental"),
        ("AFC Asian Cup", "continental"),
        ("Gold Cup", "continental"),
        ("Friendly", "friendly"),
        ("UEFA Nations League", "default"),
        ("Something random", "default"),
    ],
)
def test_classify_tournament_maps_correctly(tournament: str, expected: str) -> None:
    """`classify_tournament` associa ogni nome alla categoria K corretta."""
    assert classify_tournament(tournament) == expected


def test_classify_tournament_keys_match_k_factors_dict() -> None:
    """Ogni categoria ritornata da classify_tournament deve esistere in K_FACTORS."""
    categories = {"world_cup", "continental", "qualification", "friendly", "default"}
    assert categories == set(K_FACTORS.keys())


def test_build_history_returns_pre_match_ratings_per_row() -> None:
    """build_history aggiunge colonne home_elo_before e away_elo_before per ogni match."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "home_score": [2, 1],
            "away_score": [0, 1],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )
    elo = EloSystem()
    result = elo.build_history(matches)

    assert "home_elo_before" in result.columns
    assert "away_elo_before" in result.columns

    assert result.iloc[0]["home_elo_before"] == pytest.approx(DEFAULT_ELO, abs=0.01)
    assert result.iloc[0]["away_elo_before"] == pytest.approx(DEFAULT_ELO, abs=0.01)

    assert result.iloc[1]["home_elo_before"] > DEFAULT_ELO
    assert result.iloc[1]["away_elo_before"] == pytest.approx(DEFAULT_ELO, abs=0.01)


def test_build_history_preserves_row_order() -> None:
    """L'output ha stessa lunghezza e stesso ordine dell'input."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "home_team": ["A", "B", "C"],
            "away_team": ["B", "C", "A"],
            "home_score": [1, 2, 0],
            "away_score": [1, 0, 3],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [True, True, True],
        }
    )
    result = EloSystem().build_history(matches)

    assert len(result) == len(matches)
    pd.testing.assert_series_equal(
        result["date"].reset_index(drop=True), matches["date"].reset_index(drop=True)
    )


def test_build_history_uses_correct_k_factor_per_tournament() -> None:
    """Match in WC usa K=60, Friendly K=20, quindi gli update sono diversi."""
    matches_wc = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [1],
            "away_score": [0],
            "tournament": ["FIFA World Cup"],
            "neutral": [True],
        }
    )
    matches_friendly = matches_wc.copy()
    matches_friendly["tournament"] = ["Friendly"]

    elo_wc = EloSystem()
    elo_wc.build_history(matches_wc)

    elo_fr = EloSystem()
    elo_fr.build_history(matches_friendly)

    delta_wc = elo_wc.get("A") - DEFAULT_ELO
    delta_fr = elo_fr.get("A") - DEFAULT_ELO
    assert delta_wc > delta_fr
    assert delta_wc == pytest.approx(3 * delta_fr, abs=0.1)


def test_build_history_raises_if_not_sorted_by_date() -> None:
    """Input non ordinato per data → ValueError (protegge da data leakage sottile)."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-01"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "home_score": [1, 1],
            "away_score": [0, 0],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )
    with pytest.raises(ValueError, match="must be sorted by date ascending"):
        EloSystem().build_history(matches)


def test_elo_france_end_2018_in_plausible_range() -> None:
    """Sanity check: Elo Francia fine 2018 (post WC win) in [1950, 2200]."""
    csv_path = CONFIG.data_raw / "results.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found — run `mondiali ingest` first")

    df = load_international_results(csv_path)
    df = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2018-12-31")].copy()
    df = df.sort_values("date").reset_index(drop=True)

    elo = EloSystem()
    elo.build_history(df)

    france_elo = elo.get("France")
    assert 1950.0 <= france_elo <= 2200.0, (
        f"Francia Elo = {france_elo:.1f}, fuori range [1950, 2200] — "
        "formula Elo o K-factor potrebbero essere buggate"
    )


def test_elo_top_teams_end_2018_all_high() -> None:
    """Francia e Brasile a fine 2018 sopra 1900: top mondiali di quel periodo."""
    csv_path = CONFIG.data_raw / "results.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found")

    df = load_international_results(csv_path)
    df = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2018-12-31")].copy()
    df = df.sort_values("date").reset_index(drop=True)

    elo = EloSystem()
    elo.build_history(df)

    for team in ["France", "Brazil"]:
        assert elo.get(team) > 1900.0, f"{team} Elo = {elo.get(team):.1f}, below 1900"
