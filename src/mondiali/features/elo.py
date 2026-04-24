"""Sistema Elo custom per squadre nazionali.

K-factor variabile per importanza competizione (vedi `config.K_FACTORS`),
home advantage standard a 65 punti (zero per venue neutral).

Conforme allo spec sezione 4.4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from mondiali.config import HOME_ADVANTAGE

log = structlog.get_logger(__name__)

DEFAULT_ELO: int = 1500


def classify_tournament(tournament: str) -> str:
    """Mappa il nome del torneo alle categorie di K-factor.

    Regole (ordine di precedenza):
    1. se contiene 'qualification' → 'qualification' (batte tutto)
    2. 'FIFA World Cup' (senza qualification) → 'world_cup'
    3. Euro, Copa, AFC Asian Cup, African Cup, Gold Cup → 'continental'
    4. 'Friendly' → 'friendly'
    5. altrimenti (Nations League, tornei minori) → 'default'
    """
    t = tournament.lower()
    if "qualification" in t:
        return "qualification"
    if "fifa world cup" in t:
        return "world_cup"
    continental_keywords = (
        "uefa euro",
        "copa américa",
        "copa america",
        "african cup of nations",
        "africa cup of nations",
        "afc asian cup",
        "gold cup",
    )
    if any(kw in t for kw in continental_keywords):
        return "continental"
    if t == "friendly":
        return "friendly"
    return "default"


@dataclass
class EloSystem:
    """Elo storico in-memory. `get(team)` restituisce il rating corrente."""

    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        """Rating corrente di `team`; DEFAULT_ELO se mai visto."""
        return self.ratings.get(team, float(DEFAULT_ELO))

    def update(
        self,
        *,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        k_factor: float,
        neutral: bool,
    ) -> tuple[float, float]:
        """Applica l'update Elo per un singolo match. Zero-sum.

        Formula:
            expected_home = 1 / (1 + 10^((elo_away - elo_home_adj) / 400))
            dove elo_home_adj = elo_home + (HOME_ADVANTAGE if not neutral else 0)
            score_home = 1 if home_goals > away_goals, 0.5 if tie, 0 otherwise
            delta = k_factor * (score_home - expected_home)
            elo_home_new = elo_home + delta
            elo_away_new = elo_away - delta

        Args:
            home, away: nomi squadre.
            home_goals, away_goals: gol segnati.
            k_factor: K per questa partita.
            neutral: True se venue neutrale (disattiva home advantage).

        Returns:
            (elo_home_pre, elo_away_pre) — i rating PRIMA dell'update (utile per
            snapshot per il match stesso, dove serve il pre-match).
        """
        elo_h = self.get(home)
        elo_a = self.get(away)

        adv = 0.0 if neutral else float(HOME_ADVANTAGE)
        expected_home = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + adv)) / 400.0))

        if home_goals > away_goals:
            score_home = 1.0
        elif home_goals < away_goals:
            score_home = 0.0
        else:
            score_home = 0.5

        delta = k_factor * (score_home - expected_home)
        self.ratings[home] = elo_h + delta
        self.ratings[away] = elo_a - delta
        return elo_h, elo_a
