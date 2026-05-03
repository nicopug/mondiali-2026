"""Test scraper Transfermarkt: CDX, parsing HTML, fallback chain, cache."""
from __future__ import annotations

from datetime import date

import responses

from mondiali.data.transfermarkt import CDXRow, _query_cdx


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
