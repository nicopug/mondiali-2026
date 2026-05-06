"""Test scraper Transfermarkt: CDX, parsing HTML, fallback chain, cache."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import responses

from mondiali.data.transfermarkt import CDXRow, _parse_squad_value, _parse_value_eur, _query_cdx

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("input_str, expected", [
    # US format (2022)
    ("€80.00m", 80_000_000.0),
    ("€500k", 500_000.0),
    ("€1.50m", 1_500_000.0),
    ("€999.99k", 999_990.0),
    # German format (2014/2018)
    ("20,50 Mill. €", 20_500_000.0),
    ("5,00 Tsd. €", 5_000.0),
    ("1,50 Mio. €", 1_500_000.0),
    ("80,00 Mill. €", 80_000_000.0),
    ("75 Th. €", 75_000.0),
    # Sentinels
    ("€-", None),
    ("-", None),
    ("", None),
])
def test_parse_value_eur(input_str, expected):
    assert _parse_value_eur(input_str) == expected


@pytest.mark.parametrize("fixture_name, expected_n, expected_total, expected_top11", [
    ("tm_italy_2014.html", 24, 313_575_000.0, 218_500_000.0),
    ("tm_italy_2018.html", 26, 664_000_000.0, 416_000_000.0),
    ("tm_italy_2022.html", 29, 646_300_000.0, 439_000_000.0),
])
def test_parse_squad_value_real_fixtures(fixture_name, expected_n, expected_total, expected_top11):
    """Le 3 fixture HTML reali devono parsare ai valori esatti attesi."""
    html = (FIXTURES_DIR / fixture_name).read_text(encoding="utf-8")
    result = _parse_squad_value(html)
    assert result is not None
    assert result.n_players == expected_n
    assert result.total_value_eur == pytest.approx(expected_total, rel=1e-6)
    assert result.top11_value_eur == pytest.approx(expected_top11, rel=1e-6)
    assert result.top11_value_eur <= result.total_value_eur


def test_parse_squad_value_empty_html_returns_none():
    """HTML senza tabella rosa → None."""
    result = _parse_squad_value("<html><body>404 Not Found</body></html>")
    assert result is None


@responses.activate
def test_query_cdx_returns_parsed_rows():
    """CDX risposta JSON → lista di CDXRow."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180823120000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html",
                "200",
                "ABC123DEF",
                "12345",
            ],
        ],
    )
    rows = _query_cdx(
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        date(2018, 5, 1),
        date(2018, 9, 1),
    )
    assert len(rows) == 1
    assert isinstance(rows[0], CDXRow)
    assert rows[0].timestamp == "20180823120000"
    assert rows[0].statuscode == "200"


@responses.activate
def test_query_cdx_returns_empty_on_no_match():
    """CDX risposta vuota (solo header) → lista vuota."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
        ],
    )
    rows = _query_cdx(
        "https://www.transfermarkt.com/eritrea/startseite/verein/9999",
        date(2018, 1, 1),
        date(2018, 12, 31),
    )
    assert rows == []


@responses.activate
def test_query_cdx_returns_empty_on_404():
    """Wayback ritorna 404 (URL mai archiviato) → lista vuota."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        status=404,
    )
    rows = _query_cdx(
        "https://www.transfermarkt.com/whatever/startseite/verein/0",
        date(2018, 1, 1),
        date(2018, 12, 31),
    )
    assert rows == []


@responses.activate
def test_query_cdx_filters_to_statuscode_200():
    """Verifica che filter=statuscode:200 sia passato come query param."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]],
    )
    _query_cdx("https://example.com", date(2018, 1, 1), date(2018, 12, 31))
    call = responses.calls[0]
    assert "filter=statuscode%3A200" in call.request.url or "filter=statuscode:200" in call.request.url
