"""Configurazione globale del progetto mondiali.

Paths, costanti Elo, e parametri condivisi. Tutti i path sono risolti rispetto
alla project root (una directory sopra `src/`).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Paths(BaseModel):
    """Filesystem paths del progetto."""

    project_root: Path = Field(default_factory=lambda: _PROJECT_ROOT)
    data_raw: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "raw")
    data_processed: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "processed")
    data_manual: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "manual")
    models_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "models")
    reports_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "reports")


CONFIG = Paths()


# K-factor per aggiornamento Elo, variabile per importanza competizione.
# Valori allineati allo standard eloratings.net.
K_FACTORS: dict[str, int] = {
    "world_cup": 60,
    "continental": 50,
    "qualification": 40,
    "friendly": 20,
    "default": 30,
}

# Home advantage in punti Elo (sommato al rating casa nel calcolo expected).
# Azzerato quando is_neutral_venue=True.
HOME_ADVANTAGE: int = 65

# Random state globale per riproducibilità.
RANDOM_STATE: int = 42
