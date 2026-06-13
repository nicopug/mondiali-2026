"""Tabellone knockout ufficiale WC2026 e parsing dei bracket.

La struttura del Round of 32 e' fissa (slot per posizione di girone); solo
l'assegnazione delle 8 migliori terze dipende da quali terze si qualificano
(Annex C del regolamento FIFA, 495 combinazioni).

``OFFICIAL_R32`` e' gia' nell'**ordine a foglie** richiesto da
``simulate_knockout_bracket``: accoppia posizionalmente (0,1), (2,3), ... e i
vincenti collassano sull'albero binario fino a riprodurre il bracket ufficiale
(R16 89-96, QF 97-100, SF 101-102, Finale 104).

Slot: ``W-X`` = vincitrice girone X, ``RU-X`` = seconda girone X,
``3-XYZ`` = terza proveniente da uno dei gironi elencati.
"""
from __future__ import annotations

# Ordine a foglie (match R32 ufficiali): 74,77,73,75,83,84,81,82,76,78,79,80,86,88,85,87
OFFICIAL_R32: list[dict[str, object]] = [
    {"match": 74, "slot_a": "W-E", "slot_b": "3-ABCDF"},
    {"match": 77, "slot_a": "W-I", "slot_b": "3-CDFGH"},
    {"match": 73, "slot_a": "RU-A", "slot_b": "RU-B"},
    {"match": 75, "slot_a": "W-F", "slot_b": "RU-C"},
    {"match": 83, "slot_a": "RU-K", "slot_b": "RU-L"},
    {"match": 84, "slot_a": "W-H", "slot_b": "RU-J"},
    {"match": 81, "slot_a": "W-D", "slot_b": "3-BEFIJ"},
    {"match": 82, "slot_a": "W-G", "slot_b": "3-AEHIJ"},
    {"match": 76, "slot_a": "W-C", "slot_b": "RU-F"},
    {"match": 78, "slot_a": "RU-E", "slot_b": "RU-I"},
    {"match": 79, "slot_a": "W-A", "slot_b": "3-CEFHI"},
    {"match": 80, "slot_a": "W-L", "slot_b": "3-EHIJK"},
    {"match": 86, "slot_a": "W-J", "slot_b": "RU-H"},
    {"match": 88, "slot_a": "RU-D", "slot_b": "RU-G"},
    {"match": 85, "slot_a": "W-B", "slot_b": "3-EFGIJ"},
    {"match": 87, "slot_a": "W-K", "slot_b": "3-DEIJL"},
]

# Per un tabellone a 32 squadre: 5 turni dopo l'ingresso nel R32.
ROUND_LABELS: list[str] = ["R16", "QF", "SF", "Final", "Winner"]


class BracketError(ValueError):
    """Bracket malformato o incompleto."""


def load_bracket(data: dict) -> list[dict[str, str]]:
    """Valida e normalizza un bracket R32 da dict JSON.

    Args:
        data: dict con chiave ``bracket_r32`` = lista di 16 dict
            ``{"team_a": ..., "team_b": ...}`` in ordine a foglie
            (vedi ``OFFICIAL_R32``).

    Returns:
        lista di 16 ``{"team_a", "team_b"}`` con nomi puliti.

    Raises:
        BracketError: numero di match != 16, nome vuoto, o squadre non distinte.
    """
    pairs = data.get("bracket_r32")
    if not isinstance(pairs, list):
        raise BracketError("Manca la chiave 'bracket_r32' (lista) nel file.")
    if len(pairs) != 16:
        raise BracketError(
            f"Il bracket R32 deve avere 16 match, trovati {len(pairs)}."
        )

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, p in enumerate(pairs):
        a = str(p.get("team_a", "")).strip()
        b = str(p.get("team_b", "")).strip()
        if not a or not b:
            raise BracketError(
                f"Match #{i + 1}: nome squadra vuoto — compila tutti gli slot "
                f"del template prima di simulare."
            )
        for t in (a, b):
            if t in seen:
                raise BracketError(
                    f"Squadra duplicata '{t}': servono 32 squadre distinte."
                )
            seen.add(t)
        out.append({"team_a": a, "team_b": b})

    if len(seen) != 32:
        raise BracketError(f"Servono 32 squadre distinte, trovate {len(seen)}.")
    return out
