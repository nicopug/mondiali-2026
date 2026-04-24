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

    def build_history(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Itera sui match (ordinati per data) e ritorna df con Elo pre-match per riga.

        Colonne richieste in input: date, home_team, away_team, home_score, away_score,
        tournament, neutral.

        Output: stesse colonne + `home_elo_before`, `away_elo_before`, `k_factor_used`.

        Muta lo stato interno (`self.ratings`) con i rating finali dopo tutti i match.

        Raises:
            ValueError: se `matches` non è ordinato per data crescente.
        """
        dates = matches["date"]
        if not dates.is_monotonic_increasing:
            raise ValueError(
                "matches must be sorted by date ascending before calling build_history"
            )

        home_elo_before: list[float] = []
        away_elo_before: list[float] = []
        k_factors_used: list[int] = []

        for row in matches.itertuples(index=False):
            category = classify_tournament(row.tournament)
            k = K_FACTORS[category]
            pre_home, pre_away = self.update(
                home=row.home_team,
                away=row.away_team,
                home_goals=int(row.home_score),
                away_goals=int(row.away_score),
                k_factor=float(k),
                neutral=bool(row.neutral),
            )
            home_elo_before.append(pre_home)
            away_elo_before.append(pre_away)
            k_factors_used.append(k)

        result = matches.copy()
        result["home_elo_before"] = home_elo_before
        result["away_elo_before"] = away_elo_before
        result["k_factor_used"] = k_factors_used
        return result
