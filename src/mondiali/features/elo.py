"""Sistema Elo custom per squadre nazionali.

K-factor variabile per importanza competizione (vedi `config.K_FACTORS`),
home advantage standard a 65 punti (zero per venue neutral).

Conforme allo spec sezione 4.4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog

from mondiali.config import HOME_ADVANTAGE, K_FACTORS

log = structlog.get_logger(__name__)

DEFAULT_ELO: int = 1500


def classify_tournament(tournament: str) -> str:
    """Mappa il nome del torneo alle categorie di K-factor.

    Returns one of: 'world_cup', 'continental', 'qualification', 'friendly', 'default'.
    """
    raise NotImplementedError  # Task 7


@dataclass
class EloSystem:
    """Elo storico in-memory. `get(team)` restituisce il rating corrente."""

    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        """Rating corrente di `team`; DEFAULT_ELO se mai visto."""
        return self.ratings.get(team, float(DEFAULT_ELO))
