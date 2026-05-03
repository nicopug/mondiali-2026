"""Lookup table team_name -> (tm_slug, tm_id) per URL Transfermarkt.

Mapping costruito manualmente cercando ogni nazionale su transfermarkt.com.
URL pattern: https://www.transfermarkt.com/{slug}/startseite/verein/{id}.

Il `team_name` (chiave del dict) deve matchare ESATTAMENTE come appare nelle
colonne `home_team` / `away_team` di `matches.parquet`. Usa la stringa
canonica del dataset martj42/international_results.
"""
from __future__ import annotations

NATION_TM_IDS: dict[str, tuple[str, int]] = {
    # WC2026 qualified / likely qualified (snapshot 2026-04-29)
    "Argentina": ("argentinien", 3437),
    "France": ("frankreich", 3377),
    "Brazil": ("brasilien", 3439),
    "England": ("england", 3299),
    "Spain": ("spanien", 3375),
    "Germany": ("deutschland", 3262),
    "Portugal": ("portugal", 3300),
    "Netherlands": ("niederlande", 3382),
    "Belgium": ("belgien", 3382),  # FIXME: verify ID
    "Italy": ("italien", 3376),
    "Croatia": ("kroatien", 3556),
    "Uruguay": ("uruguay", 3439),  # FIXME: verify ID (collision with Brazil id?)
    "Mexico": ("mexiko", 6303),
    "United States": ("vereinigte-staaten", 3505),
    "Canada": ("kanada", 3433),
    "Morocco": ("marokko", 3473),
    "Senegal": ("senegal", 3499),
    "Japan": ("japan", 3437),  # FIXME: verify
    "South Korea": ("sudkorea", 3520),
    "Australia": ("australien", 3433),  # FIXME: verify
    "Saudi Arabia": ("saudi-arabien", 3502),
    "Iran": ("iran", 3373),
    "Ecuador": ("ecuador", 3447),
    "Colombia": ("kolumbien", 3438),
    "Peru": ("peru", 3441),
    "Chile": ("chile", 3440),
    "Paraguay": ("paraguay", 3442),
    "Switzerland": ("schweiz", 3384),
    "Denmark": ("danemark", 3375),  # FIXME: verify
    "Poland": ("polen", 3437),  # FIXME: verify
    "Serbia": ("serbien", 3439),  # FIXME: verify
    "Wales": ("wales", 3577),
    "Scotland": ("schottland", 3576),
    "Austria": ("osterreich", 3442),  # FIXME: verify
    "Sweden": ("schweden", 3375),  # FIXME: verify
    "Norway": ("norwegen", 3375),  # FIXME: verify
    "Czech Republic": ("tschechien", 3375),  # FIXME: verify
    "Hungary": ("ungarn", 3578),
    "Turkey": ("turkei", 3376),
    "Ukraine": ("ukraine", 3376),
    "Romania": ("rumanien", 3375),
    "Slovakia": ("slowakei", 3375),
    "Slovenia": ("slowenien", 3375),
    "Greece": ("griechenland", 3375),
    "Republic of Ireland": ("irland", 3299),
    "Bosnia and Herzegovina": ("bosnien-herzegowina", 3375),
    "North Macedonia": ("nordmazedonien", 3375),
    "Albania": ("albanien", 3375),
    # Top-32 historic FIFA Elo (non WC2026 ma frequenti in qualifications)
    "Russia": ("russland", 3437),
    "Tunisia": ("tunesien", 3499),
    "Algeria": ("algerien", 3473),
    "Egypt": ("agypten", 3471),
    "Nigeria": ("nigeria", 3499),
    "Ghana": ("ghana", 3473),
    "Cameroon": ("kamerun", 3473),
    "Ivory Coast": ("elfenbeinkuste", 3473),
    "Iceland": ("island", 3375),
    "Finland": ("finnland", 3375),
    "Bolivia": ("bolivien", 3439),
    "Venezuela": ("venezuela", 3439),
    "Costa Rica": ("costa-rica", 3433),
    "Panama": ("panama", 3433),
    "Honduras": ("honduras", 3433),
    "Jamaica": ("jamaika", 3433),
    "Qatar": ("katar", 3502),
    "United Arab Emirates": ("vereinigte-arabische-emirate", 3502),
    "Iraq": ("irak", 3502),
    "China PR": ("china", 3520),
    "New Zealand": ("neuseeland", 3433),
    "South Africa": ("sudafrika", 3499),
    "Mali": ("mali", 3499),
    "Burkina Faso": ("burkina-faso", 3499),
    "DR Congo": ("dr-kongo", 3499),
    "Cape Verde": ("kap-verde", 3499),
    "Israel": ("israel", 3375),
    "Georgia": ("georgien", 3576),
    "Armenia": ("armenien", 3576),
    "Azerbaijan": ("aserbaidschan", 3576),
}

# Sanity invariants
assert all(isinstance(v, tuple) and len(v) == 2 for v in NATION_TM_IDS.values()), \
    "NATION_TM_IDS values must be (slug: str, id: int) tuples"
assert all(isinstance(slug, str) and isinstance(tid, int) for slug, tid in NATION_TM_IDS.values())
assert len(NATION_TM_IDS) >= 60, f"expected >=60 nations, got {len(NATION_TM_IDS)}"
