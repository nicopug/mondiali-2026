"""Discovery dei TM IDs reali via schnellsuche live (NON Wayback).

Hotfix per `tm_nations.py` (bootstrap aveva ~63/78 ID collidenti).
Strategia: query TM search col nome inglese, filtra gli href
``/{slug}/startseite/verein/{id}`` per slug match esatto col bootstrap.
Lo SLUG del bootstrap è considerato verità (verificato manualmente);
solo l'ID viene riscoperto.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests
import structlog

from mondiali.data.tm_nations import NATION_TM_IDS

log = structlog.get_logger(__name__)

TM_SEARCH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
RATE_LIMIT_SECONDS = 1.5
SEARCH_TIMEOUT_SECONDS = 30.0

_HREF_RE = re.compile(r'href="/([a-z0-9-]+)/startseite/verein/(\d+)')
_EXCLUDE_SUFFIX = re.compile(r"-(u\d+|olympia|frauen|junioren|amateure)\b")


def parse_team_id(html: str, expected_slug: str) -> int | None:
    """Estrai l'ID TM associato allo slug dato dai risultati search.

    Strategia in due passi:
    1. Match ESATTO sullo slug (preferito).
    2. Match ``slug-something`` ESCLUDENDO suffissi di età/genere.
    """
    if not html:
        return None
    hits = _HREF_RE.findall(html)
    for slug, tm_id in hits:
        if slug == expected_slug:
            return int(tm_id)
    for slug, tm_id in hits:
        if slug.startswith(expected_slug + "-") and not _EXCLUDE_SUFFIX.search(slug):
            return int(tm_id)
    return None


def fetch_search_html(query: str, *, timeout: float = SEARCH_TIMEOUT_SECONDS) -> str | None:
    """GET schnellsuche?query=... con UA browser-like. Ritorna None su errore."""
    try:
        resp = requests.get(
            TM_SEARCH_URL,
            params={"query": query},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        log.warning("tm_search_exception", query=query, error=str(exc))
        return None
    if resp.status_code != 200:
        log.warning("tm_search_non200", query=query, status=resp.status_code)
        return None
    return resp.text


def discover_all_team_ids() -> dict[str, tuple[str, int]]:
    """Itera NATION_TM_IDS, recupera ID reale per ogni slug.

    Ritorna nuovo dict (slug invariato; id aggiornato se trovato, altrimenti
    eredita il vecchio con warning).
    """
    out: dict[str, tuple[str, int]] = {}
    n_changed = 0
    n_unchanged = 0
    n_failed = 0
    for nation, (slug, old_id) in NATION_TM_IDS.items():
        time.sleep(RATE_LIMIT_SECONDS)
        html = fetch_search_html(nation)
        new_id: int | None = parse_team_id(html, slug) if html else None
        if new_id is None and html is not None:
            # fallback: query col nome tedesco (capitalize dello slug)
            time.sleep(RATE_LIMIT_SECONDS)
            html = fetch_search_html(slug.capitalize())
            if html:
                new_id = parse_team_id(html, slug)
        if new_id is None:
            log.warning("tm_discover_failed", nation=nation, slug=slug, kept_old=old_id)
            out[nation] = (slug, old_id)
            n_failed += 1
            continue
        if new_id != old_id:
            log.info("tm_discover_updated", nation=nation, slug=slug, old=old_id, new=new_id)
            n_changed += 1
        else:
            n_unchanged += 1
        out[nation] = (slug, new_id)
    log.info(
        "tm_discover_complete",
        total=len(NATION_TM_IDS),
        changed=n_changed,
        unchanged=n_unchanged,
        failed=n_failed,
    )
    return out


_FILE_TEMPLATE = '''"""Lookup table team_name -> (tm_slug, tm_id) per URL Transfermarkt.

URL pattern: https://www.transfermarkt.com/{{slug}}/startseite/verein/{{id}}.

⚙ Generato da `mondiali tm-discover-ids` (TM live schnellsuche).
NON modificare a mano: rilancia il comando per rigenerare.
"""
from __future__ import annotations

NATION_TM_IDS: dict[str, tuple[str, int]] = {{
{entries}
}}

_BOOTSTRAP_VERIFIED: bool = True

assert all(isinstance(v, tuple) and len(v) == 2 for v in NATION_TM_IDS.values()), \\
    "NATION_TM_IDS values must be (slug: str, id: int) tuples"
assert all(isinstance(slug, str) and isinstance(tid, int) for slug, tid in NATION_TM_IDS.values()), \\
    "All values must be (str, int) tuples — check for quoted tm_id"
assert len(NATION_TM_IDS) >= 60, f"expected >=60 nations, got {{len(NATION_TM_IDS)}}"
'''


def rewrite_nations_file(mapping: dict[str, tuple[str, int]], path: Path) -> None:
    """Riscrive `tm_nations.py` con il nuovo mapping (slug invariato, id aggiornato)."""
    lines = []
    for nation, (slug, tm_id) in mapping.items():
        nation_repr = repr(nation)
        slug_repr = repr(slug)
        lines.append(f"    {nation_repr}: ({slug_repr}, {tm_id}),")
    content = _FILE_TEMPLATE.format(entries="\n".join(lines))
    path.write_text(content, encoding="utf-8")
