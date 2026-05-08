"""Lookup table team_name -> (tm_slug, tm_id) per URL Transfermarkt.

URL pattern: https://www.transfermarkt.com/{slug}/startseite/verein/{id}.

⚙ Generato da `mondiali tm-discover-ids` (TM live schnellsuche).
NON modificare a mano: rilancia il comando per rigenerare.
"""
from __future__ import annotations

NATION_TM_IDS: dict[str, tuple[str, int]] = {
    'Argentina': ('argentinien', 3437),
    'France': ('frankreich', 3377),
    'Brazil': ('brasilien', 3439),
    'England': ('england', 3299),
    'Spain': ('spanien', 3375),
    'Germany': ('deutschland', 3262),
    'Portugal': ('portugal', 3300),
    'Netherlands': ('niederlande', 3379),
    'Belgium': ('belgien', 3382),
    'Italy': ('italien', 3376),
    'Croatia': ('kroatien', 3556),
    'Uruguay': ('uruguay', 3449),
    'Mexico': ('mexiko', 6303),
    'United States': ('vereinigte-staaten', 3505),
    'Canada': ('kanada', 3510),
    'Morocco': ('marokko', 3575),
    'Senegal': ('senegal', 3499),
    'Japan': ('japan', 3435),
    'South Korea': ('sudkorea', 3589),
    'Australia': ('australien', 3433),
    'Saudi Arabia': ('saudi-arabien', 3807),
    'Iran': ('iran', 3582),
    'Ecuador': ('ecuador', 5750),
    'Colombia': ('kolumbien', 3816),
    'Peru': ('peru', 3584),
    'Chile': ('chile', 3700),
    'Paraguay': ('paraguay', 3581),
    'Switzerland': ('schweiz', 3384),
    'Denmark': ('danemark', 3436),
    'Poland': ('polen', 3442),
    'Serbia': ('serbien', 3438),
    'Wales': ('wales', 3864),
    'Scotland': ('schottland', 3380),
    'Austria': ('osterreich', 3383),
    'Sweden': ('schweden', 3557),
    'Norway': ('norwegen', 3440),
    'Czech Republic': ('tschechien', 3445),
    'Hungary': ('ungarn', 3468),
    'Turkey': ('turkei', 3381),
    'Ukraine': ('ukraine', 3699),
    'Romania': ('rumanien', 3447),
    'Slovakia': ('slowakei', 3503),
    'Slovenia': ('slowenien', 3588),
    'Greece': ('griechenland', 3378),
    'Republic of Ireland': ('irland', 3509),
    'Bosnia and Herzegovina': ('bosnien-herzegowina', 3446),
    'North Macedonia': ('nordmazedonien', 5148),
    'Albania': ('albanien', 3561),
    'Russia': ('russland', 3448),
    'Tunisia': ('tunesien', 3670),
    'Algeria': ('algerien', 3614),
    'Egypt': ('agypten', 3672),
    'Nigeria': ('nigeria', 3444),
    'Ghana': ('ghana', 3441),
    'Cameroon': ('kamerun', 3434),
    'Ivory Coast': ('elfenbeinkuste', 3591),
    'Iceland': ('island', 3574),
    'Finland': ('finnland', 3443),
    'Bolivia': ('bolivien', 5233),
    'Venezuela': ('venezuela', 3504),
    'Costa Rica': ('costa-rica', 8497),
    'Panama': ('panama', 3577),
    'Honduras': ('honduras', 3590),
    'Jamaica': ('jamaika', 3671),
    'Qatar': ('katar', 14162),
    'United Arab Emirates': ('vereinigte-arabische-emirate', 5147),
    'Iraq': ('irak', 3560),
    'China PR': ('china', 5598),
    'New Zealand': ('neuseeland', 9171),
    'South Africa': ('sudafrika', 3806),
    'Mali': ('mali', 3674),
    'Burkina Faso': ('burkina-faso', 5872),
    'DR Congo': ('demokratische-republik-kongo', 3854),
    'Cape Verde': ('kap-verde', 4311),
    'Israel': ('israel', 5547),
    'Georgia': ('georgien', 3669),
    'Armenia': ('armenien', 6219),
    'Azerbaijan': ('aserbaidschan', 8605),
}

_BOOTSTRAP_VERIFIED: bool = True

assert all(isinstance(v, tuple) and len(v) == 2 for v in NATION_TM_IDS.values()), \
    "NATION_TM_IDS values must be (slug: str, id: int) tuples"
assert all(isinstance(slug, str) and isinstance(tid, int) for slug, tid in NATION_TM_IDS.values()), \
    "All values must be (str, int) tuples — check for quoted tm_id"
assert len(NATION_TM_IDS) >= 60, f"expected >=60 nations, got {len(NATION_TM_IDS)}"
