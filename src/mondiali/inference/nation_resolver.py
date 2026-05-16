"""Resolve user nation inputs to canonical names known by the model.

Supports:
- Exact match (case-insensitive)
- Common aliases ("USA" -> "United States", "UK" -> "England", ...)
- Levenshtein fuzzy matching via stdlib difflib

If no match is found within MIN_SIMILARITY, raises NationNotFound with the
top 3 suggestions.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

# Common aliases users might type. Canonical names come from matches.parquet's
# home_team/away_team columns; we map informal inputs to them.
ALIASES: dict[str, str] = {
    "usa": "United States",
    "us": "United States",
    "u.s.a.": "United States",
    "uk": "England",  # debatable but typical user intent
    "great britain": "England",
    "uae": "United Arab Emirates",
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "czechia": "Czech Republic",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "curacao": "Curaçao",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "republic of ireland": "Republic of Ireland",
    "ireland": "Republic of Ireland",
    "russia": "Russia",
    "iran": "Iran",
    "drc": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "congo dr": "DR Congo",
    "north macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "swaziland": "Eswatini",
    "burma": "Myanmar",
    "cape verde": "Cape Verde",
    "cabo verde": "Cape Verde",
    "east timor": "Timor-Leste",
    "vatican": "Vatican City",
    "kyrgyzstan": "Kyrgyzstan",
    "kirghizistan": "Kyrgyzstan",
    "kazakhstan": "Kazakhstan",
    "trinidad": "Trinidad and Tobago",
    "tobago": "Trinidad and Tobago",
    "antigua": "Antigua and Barbuda",
    "saint kitts": "Saint Kitts and Nevis",
    "saint vincent": "Saint Vincent and the Grenadines",
    "saint lucia": "Saint Lucia",
    "central african republic": "Central African Republic",
    "car": "Central African Republic",
    "saudi": "Saudi Arabia",
    "saudi arabia": "Saudi Arabia",
}

MIN_SIMILARITY = 0.7
N_SUGGESTIONS = 3


class NationNotFound(ValueError):
    """Raised when input cannot be resolved to a known nation."""

    def __init__(self, query: str, suggestions: list[str]) -> None:
        self.query = query
        self.suggestions = suggestions
        msg = f"Unknown nation '{query}'."
        if suggestions:
            msg += f" Did you mean: {', '.join(repr(s) for s in suggestions)}?"
        else:
            msg += " No close match found."
        super().__init__(msg)


@dataclass
class NationResolver:
    """Resolves nation inputs against a fixed corpus of canonical names."""

    canonical_names: list[str]

    @classmethod
    def from_state_dir(cls, state_dir) -> NationResolver:
        import pandas as pd
        from pathlib import Path
        path = Path(state_dir) / "elo_state.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"elo_state.parquet not found at {path}; run `mondiali update-state` first"
            )
        names = pd.read_parquet(path)["nation"].astype(str).tolist()
        return cls(canonical_names=sorted(set(names)))

    def resolve(self, query: str) -> str:
        """Return canonical name or raise NationNotFound with suggestions."""
        q = query.strip()
        if not q:
            raise NationNotFound(query, [])

        # 1. Exact match (case-sensitive)
        canonical_lower = {n.lower(): n for n in self.canonical_names}
        if q in self.canonical_names:
            return q
        # 2. Case-insensitive exact
        if q.lower() in canonical_lower:
            return canonical_lower[q.lower()]
        # 3. Alias map
        if q.lower() in ALIASES:
            mapped = ALIASES[q.lower()]
            if mapped in self.canonical_names:
                return mapped
        # 4. Fuzzy match
        suggestions = get_close_matches(
            q, self.canonical_names, n=N_SUGGESTIONS, cutoff=MIN_SIMILARITY,
        )
        # Also try matching against alias keys
        alias_matches = get_close_matches(
            q.lower(), list(ALIASES.keys()), n=N_SUGGESTIONS, cutoff=MIN_SIMILARITY,
        )
        for am in alias_matches:
            target = ALIASES[am]
            if target in self.canonical_names and target not in suggestions:
                suggestions.append(target)
        suggestions = suggestions[:N_SUGGESTIONS]
        raise NationNotFound(query, suggestions)
