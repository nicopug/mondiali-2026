"""Test scraper Transfermarkt: CDX, parsing HTML, fallback chain, cache."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import json

import pandas as pd
import pytest
import responses

from mondiali.data.transfermarkt import (
    CDXRow,
    _best_snapshot_for_year,
    _fetch_snapshot_html,
    _parse_squad_value,
    parse_value_eur,
    _query_cdx,
    _wayback_url,
    build_from_cache,
    scrape_all,
)

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
    # Unsupported suffixes (bn not implemented, only m/k/Mio/Mill/Tsd/Th)
    ("€1.20bn", None),
    # Defensive: random suffix should not match
    ("€100xyz", None),
])
def test_parse_value_eur(input_str, expected):
    assert parse_value_eur(input_str) == expected


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


def test_wayback_url_construction():
    row = CDXRow(
        urlkey="com,transfermarkt)/italien/startseite/verein/3376",
        timestamp="20180823120000",
        original="https://www.transfermarkt.com/italien/startseite/verein/3376",
        mimetype="text/html",
        statuscode="200",
        digest="ABC",
        length="123",
    )
    url = _wayback_url(row)
    assert url == "https://web.archive.org/web/20180823120000/https://www.transfermarkt.com/italien/startseite/verein/3376"


@responses.activate
def test_fetch_snapshot_html_uses_cache(tmp_path, monkeypatch):
    """Se il file è già in cache, no HTTP call. Idempotenza."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    cache_file = tmp_path / "italien__20180823120000.html"
    cache_file.write_text("<html>cached</html>", encoding="utf-8")

    row = CDXRow(
        "com,transfermarkt)/italien/startseite/verein/3376",
        "20180823120000",
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        "text/html", "200", "ABC", "123",
    )
    html = _fetch_snapshot_html(row, tmp_path)
    assert html == "<html>cached</html>"
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_snapshot_html_writes_cache(tmp_path, monkeypatch):
    """Cache miss → fetch → write to disk."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180823120000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body="<html>fetched</html>",
        status=200,
    )
    row = CDXRow(
        "com,transfermarkt)/italien/startseite/verein/3376",
        "20180823120000",
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        "text/html", "200", "ABC", "123",
    )
    html = _fetch_snapshot_html(row, tmp_path)
    assert html == "<html>fetched</html>"
    expected_file = tmp_path / "italien__20180823120000.html"
    assert expected_file.exists()
    assert expected_file.read_text(encoding="utf-8") == "<html>fetched</html>"


@responses.activate
def test_fetch_snapshot_html_returns_none_on_5xx(tmp_path, monkeypatch):
    """HTTP 500 → retry exp-backoff esauriti → None."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    monkeypatch.setattr("mondiali.data.transfermarkt._RETRY_BACKOFF_BASE", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180823120000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        status=500,
    )
    row = CDXRow(
        "com,transfermarkt)/italien/startseite/verein/3376",
        "20180823120000",
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        "text/html", "200", "ABC", "123",
    )
    html = _fetch_snapshot_html(row, tmp_path)
    assert html is None
    assert len(responses.calls) == 3


@responses.activate
def test_best_snapshot_for_year_level1_success(tmp_path, monkeypatch):
    """Trova snapshot al primo livello (target ±60 giorni)."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180815000000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html", "200", "ABC", "12345",
            ],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180502000000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html", "200", "ABC", "12345",
            ],
        ],
    )
    fixture = (FIXTURES_DIR / "tm_italy_2018.html").read_text(encoding="utf-8")
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180815000000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body=fixture,
        status=200,
    )
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180502000000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body=fixture,
        status=200,
    )
    snap = _best_snapshot_for_year(
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        2018,
        tmp_path,
    )
    assert snap is not None
    snap_date, parsed, source = snap
    assert snap_date == date(2018, 8, 15)
    assert parsed.total_value_eur > 0
    assert source.startswith("https://web.archive.org/web/20180815000000/")


@responses.activate
def test_best_snapshot_for_year_returns_none_when_all_levels_empty(tmp_path, monkeypatch):
    """Tutti e 3 i livelli ritornano CDX vuoto → None."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]],
    )
    snap = _best_snapshot_for_year(
        "https://www.transfermarkt.com/eritrea/startseite/verein/9999",
        2018,
        tmp_path,
    )
    assert snap is None


@responses.activate
def test_best_snapshot_for_year_falls_through_to_level2(tmp_path, monkeypatch):
    """Livello 1 ritorna vuoto, livello 2 trova snapshot a marzo."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)

    # Use a callback so we can return different bodies based on call order.
    cdx_responses = [
        # level 1: empty
        [["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]],
        # level 2: returns one row in march (outside ±60d window)
        [
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180315000000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html", "200", "ABC", "12345",
            ],
        ],
    ]

    call_idx = {"i": 0}

    def cdx_callback(request):
        i = call_idx["i"]
        call_idx["i"] += 1
        return (200, {}, json.dumps(cdx_responses[min(i, len(cdx_responses) - 1)]))

    responses.add_callback(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        callback=cdx_callback,
        content_type="application/json",
    )
    fixture = (FIXTURES_DIR / "tm_italy_2018.html").read_text(encoding="utf-8")
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180315000000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body=fixture,
        status=200,
    )
    snap = _best_snapshot_for_year(
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        2018,
        tmp_path,
    )
    assert snap is not None
    snap_date, _, _ = snap
    assert snap_date == date(2018, 3, 15)


def test_best_snapshot_for_year_uses_cache_without_network(tmp_path, monkeypatch):
    """Fast path: se cache locale ha snapshot validi, salta CDX (zero requests)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fixture = (FIXTURES_DIR / "tm_italy_2018.html").read_text(encoding="utf-8")
    (cache_dir / "italien__20180815000000.html").write_text(fixture, encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("network must not be called when cache hit is available")

    monkeypatch.setattr("mondiali.data.transfermarkt.requests.get", _boom)

    snap = _best_snapshot_for_year(
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        2018,
        cache_dir,
    )
    assert snap is not None
    snap_date, parsed, source = snap
    assert snap_date == date(2018, 8, 15)
    assert parsed.total_value_eur > 0
    assert source.startswith("https://web.archive.org/web/20180815000000/")


def test_best_snapshot_for_year_cache_falls_back_to_year_minus_1(tmp_path, monkeypatch):
    """Cache ha solo file dell'anno precedente (semestre 2) → livello 3 vince."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fixture = (FIXTURES_DIR / "tm_italy_2018.html").read_text(encoding="utf-8")
    (cache_dir / "italien__20171015000000.html").write_text(fixture, encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("network must not be called when cache hit is available")

    monkeypatch.setattr("mondiali.data.transfermarkt.requests.get", _boom)

    snap = _best_snapshot_for_year(
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        2018,
        cache_dir,
    )
    assert snap is not None
    snap_date, _, _ = snap
    assert snap_date == date(2017, 10, 15)


def test_best_snapshot_for_year_cache_ignores_other_slugs(tmp_path, monkeypatch):
    """File cache di altri slug non devono essere considerati hit."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fixture = (FIXTURES_DIR / "tm_italy_2018.html").read_text(encoding="utf-8")
    # cache popolata SOLO con frankreich, query è su italien → no hit
    (cache_dir / "frankreich__20180815000000.html").write_text(fixture, encoding="utf-8")

    # CDX query deve partire (no cache hit) → mock vuoto → None
    @responses.activate
    def _run():
        responses.add(
            responses.GET,
            "https://web.archive.org/cdx/search/cdx",
            json=[["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]],
        )
        return _best_snapshot_for_year(
            "https://www.transfermarkt.com/italien/startseite/verein/3376",
            2018,
            cache_dir,
        )

    assert _run() is None


def test_build_from_cache_writes_parquet_without_network(tmp_path, monkeypatch):
    """build_from_cache: zero requests, parquet generato dai cache file esistenti."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fixture = (FIXTURES_DIR / "tm_italy_2018.html").read_text(encoding="utf-8")
    # cache popolata per Italy 2018 e 2019 (stesso fixture, ts diversi)
    (cache_dir / "italien__20180815000000.html").write_text(fixture, encoding="utf-8")
    (cache_dir / "italien__20190820000000.html").write_text(fixture, encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("network must not be called by build_from_cache")

    monkeypatch.setattr("mondiali.data.transfermarkt.requests.get", _boom)

    out = tmp_path / "snapshots.parquet"
    n_target, n_filled = build_from_cache(
        scope=["Italy"],
        years=[2018, 2019, 2024],  # 2024 fuori da fallback (richiede ≥2023 sem.2)
        cache_dir=cache_dir,
        output_path=out,
    )
    assert n_target == 3
    assert n_filled == 2
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert set(df["year"].tolist()) == {2018, 2019}
    assert df.iloc[0]["nation"] == "Italy"


@responses.activate
def test_scrape_all_resume_skips_already_done_nations(tmp_path, monkeypatch):
    """resume=True: le nazioni già in parquet vengono saltate (gap inclusi)."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)

    # Pre-popola parquet con 2 record per "Italy"
    out = tmp_path / "snapshots.parquet"
    pre_df = pd.DataFrame([
        {"nation": "Italy", "year": 2018, "snapshot_date": pd.Timestamp("2018-07-15"),
         "total_value_eur": 800_000_000.0, "top11_value_eur": 600_000_000.0,
         "n_players": 23, "source_url": "https://web.archive.org/web/foo"},
        {"nation": "Italy", "year": 2019, "snapshot_date": pd.Timestamp("2019-07-10"),
         "total_value_eur": 850_000_000.0, "top11_value_eur": 620_000_000.0,
         "n_players": 24, "source_url": "https://web.archive.org/web/bar"},
    ])
    pre_df.to_parquet(out, index=False)

    # CDX mock: deve NON essere chiamato per Italy
    cdx_calls = {"n": 0}

    def cdx_callback(request):
        cdx_calls["n"] += 1
        return (200, {}, json.dumps([
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]
        ]))

    responses.add_callback(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        callback=cdx_callback,
        content_type="application/json",
    )

    scrape_all(
        scope=["Italy"],
        years=[2018, 2019, 2020],
        cache_dir=tmp_path / "cache",
        output_path=out,
        resume=True,
    )

    # Italy doveva essere completamente skippata
    assert cdx_calls["n"] == 0, "Italy era già in parquet, non dovevano partire query CDX"

    # Output preserva i 2 record originali
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert set(df["year"]) == {2018, 2019}


@responses.activate
def test_scrape_all_writes_parquet(tmp_path, monkeypatch):
    """End-to-end: scope di 1 nazionale × 2 anni → snapshots.parquet con righe."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    fixture = (FIXTURES_DIR / "tm_italy_2018.html").read_text(encoding="utf-8")

    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180701000000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html", "200", "ABC", "12345",
            ],
        ],
    )
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180701000000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body=fixture,
        status=200,
    )
    out = tmp_path / "snapshots.parquet"
    scrape_all(
        scope=["Italy"],
        years=[2018, 2019],
        cache_dir=tmp_path / "cache",
        output_path=out,
    )
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2  # both 2018 and 2019 reuse the same mocked CDX/Wayback response
    assert set(df.columns) >= {
        "nation", "year", "snapshot_date", "total_value_eur",
        "top11_value_eur", "n_players", "source_url",
    }
    assert df.iloc[0]["nation"] == "Italy"
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])
