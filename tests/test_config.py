"""Test del modulo config: paths, K-factors, home advantage."""
from pathlib import Path

from mondiali.config import CONFIG, HOME_ADVANTAGE, K_FACTORS


def test_paths_are_absolute_and_project_scoped() -> None:
    """I path della config devono essere risolti e puntare dentro la project root."""
    assert CONFIG.data_raw.is_absolute()
    assert CONFIG.data_processed.is_absolute()
    assert CONFIG.models_dir.is_absolute()
    assert CONFIG.reports_dir.is_absolute()
    project_root = Path(__file__).parent.parent.resolve()
    for p in [CONFIG.data_raw, CONFIG.data_processed, CONFIG.models_dir, CONFIG.reports_dir]:
        assert str(p).startswith(str(project_root)), f"{p} is outside project root"


def test_k_factors_cover_all_tournament_categories() -> None:
    """K-factors devono coprire World Cup, continental, qualification, friendly, default."""
    assert K_FACTORS["world_cup"] == 60
    assert K_FACTORS["continental"] == 50
    assert K_FACTORS["qualification"] == 40
    assert K_FACTORS["friendly"] == 20
    assert K_FACTORS["default"] == 30


def test_home_advantage_standard_value() -> None:
    """Home advantage deve essere il valore standard eloratings.net."""
    assert HOME_ADVANTAGE == 65
