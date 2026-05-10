"""Tests for tm_rosters.py — player-level roster scraper."""
from __future__ import annotations

from pathlib import Path

import pytest

from mondiali.data.tm_rosters import RosterPlayer, _parse_roster_html


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_roster_html_extracts_players() -> None:
    html = (FIXTURE_DIR / "tm_roster_france_wc2018.html").read_text(encoding="utf-8")
    players = _parse_roster_html(html)
    assert len(players) == 5
    assert players[0] == RosterPlayer(
        player_name="Hugo Lloris",
        player_url_slug="hugo-lloris",
        position="Goalkeeper",
        market_value_eur=10_000_000,
    )
    mbappe = next(p for p in players if p.player_name == "Kylian Mbappé")
    assert mbappe.market_value_eur == 120_000_000
    kante = next(p for p in players if p.player_url_slug == "n-golo-kante")
    assert kante.market_value_eur == 60_000_000


def test_parse_value_handles_em_billions() -> None:
    """Sanity: TM occasionally uses bn for a national team (rare). Should not crash."""
    html = """<html><body><table class="items"><tbody>
    <tr><td class="hauptlink"><a href="/x/profil/spieler/1" title="X">X</a></td>
    <td>GK</td><td class="rechts hauptlink">€1.20bn</td></tr>
    </tbody></table></body></html>"""
    # 1.20bn is currently not supported by _parse_value_eur (only m, k, Mio, Tsd, Th).
    # We assert that the parser does NOT crash and returns None for value.
    players = _parse_roster_html(html)
    assert len(players) == 1
    assert players[0].market_value_eur is None


def test_parse_skips_rows_without_player_link() -> None:
    """Header rows or aggregate rows with no /profil/spieler/ link must be ignored."""
    html = """<html><body><table class="items"><tbody>
    <tr><td colspan="5">Total: €500.00m</td></tr>
    </tbody></table></body></html>"""
    assert _parse_roster_html(html) == []


def test_omonimi_disambiguati_via_slug() -> None:
    """Two players named 'Diego López' must produce two distinct slugs."""
    html = """<html><body><table class="items"><tbody>
    <tr><td class="hauptlink"><a href="/diego-lopez/profil/spieler/100" title="Diego López">Diego López</a></td>
    <td>GK</td><td class="rechts hauptlink">€5.00m</td></tr>
    <tr><td class="hauptlink"><a href="/diego-lopez-2/profil/spieler/200" title="Diego López">Diego López</a></td>
    <td>DEF</td><td class="rechts hauptlink">€3.00m</td></tr>
    </tbody></table></body></html>"""
    players = _parse_roster_html(html)
    slugs = {p.player_url_slug for p in players}
    assert slugs == {"diego-lopez", "diego-lopez-2"}


def test_roster_url_uses_saison_year_minus_one() -> None:
    """WC2018 → saison_id 2017 (TM stores roster under start-of-season year)."""
    from mondiali.data.tm_rosters import _build_roster_url
    url = _build_roster_url("France", "wc2018")
    assert url is not None
    assert "/saison_id/2017" in url
    assert "/kader/verein/" in url


def test_roster_url_returns_none_for_unknown_nation() -> None:
    from mondiali.data.tm_rosters import _build_roster_url
    assert _build_roster_url("Atlantis", "wc2018") is None


def test_roster_url_returns_none_for_unknown_tournament() -> None:
    from mondiali.data.tm_rosters import _build_roster_url
    assert _build_roster_url("France", "wc1990") is None


def test_cache_fast_path_returns_html_when_present(tmp_path: Path) -> None:
    """If `{slug}__{tournament}.html` exists, _read_cached_roster reads it without network."""
    from mondiali.data.tm_rosters import _read_cached_roster
    cache_dir = tmp_path / "rosters"
    cache_dir.mkdir()
    (cache_dir / "equipe-de-france__wc2018.html").write_text("<html>cached</html>", encoding="utf-8")
    out = _read_cached_roster("equipe-de-france", "wc2018", cache_dir)
    assert out == "<html>cached</html>"


def test_cache_fast_path_returns_none_when_missing(tmp_path: Path) -> None:
    from mondiali.data.tm_rosters import _read_cached_roster
    assert _read_cached_roster("nope", "wc2018", tmp_path) is None
