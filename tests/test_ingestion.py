"""Test per data ingestion: download e parsing di international_results."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mondiali.data.ingestion import (
    INTERNATIONAL_RESULTS_URL,
    build_processed_matches,
    download_international_results,
    load_international_results,
)
from mondiali.features.tier3 import TIER3_COLUMNS


def test_download_writes_csv_to_destination(tmp_path: Path) -> None:
    """Il download scrive il CSV alla destinazione specificata."""
    dest = tmp_path / "results.csv"
    fake_csv = (
        b"date,home_team,away_team,home_score,away_score,"
        b"tournament,city,country,neutral\n"
        b"1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE\n"
    )

    with patch("mondiali.data.ingestion.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = fake_csv
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result_path = download_international_results(dest)

    assert result_path == dest
    assert dest.exists()
    assert dest.read_bytes() == fake_csv
    mock_get.assert_called_once_with(INTERNATIONAL_RESULTS_URL, timeout=60)


def test_download_skips_if_file_exists_and_force_false(tmp_path: Path) -> None:
    """Se il file esiste e force=False, non ri-scarica."""
    dest = tmp_path / "results.csv"
    dest.write_bytes(b"existing content")

    with patch("mondiali.data.ingestion.requests.get") as mock_get:
        download_international_results(dest, force=False)

    mock_get.assert_not_called()
    assert dest.read_bytes() == b"existing content"


def test_download_raises_on_http_error(tmp_path: Path) -> None:
    """HTTP error propagato correttamente."""
    dest = tmp_path / "results.csv"

    with patch("mondiali.data.ingestion.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = RuntimeError("HTTP 500")
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="HTTP 500"):
            download_international_results(dest)


def test_load_parses_dates_and_normalizes_columns(tmp_path: Path) -> None:
    """Parsing del CSV produce DataFrame con date pandas e tipi coerenti."""
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE\n"
    )

    df = load_international_results(csv)

    assert list(df.columns) == [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
    assert df["date"].dtype == "datetime64[ns]"
    assert df["home_score"].dtype == "int64"
    assert df["away_score"].dtype == "int64"
    assert df["neutral"].dtype == "bool"
    assert len(df) == 2
    assert df.iloc[0]["home_team"] == "France"
    assert bool(df.iloc[0]["neutral"]) is True


def test_load_drops_rows_with_missing_scores(tmp_path: Path) -> None:
    """Match senza punteggio (future o cancellati) sono droppati."""
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2026-06-11,Mexico,USA,,,FIFA World Cup,,USA,FALSE\n"
    )

    df = load_international_results(csv)

    assert len(df) == 1
    assert df.iloc[0]["home_team"] == "France"


def test_load_sorts_by_date_ascending(tmp_path: Path) -> None:
    """Rows ordinate per data crescente."""
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
    )

    df = load_international_results(csv)

    assert df.iloc[0]["date"] < df.iloc[1]["date"]


def test_build_processed_matches_produces_expected_schema(tmp_path: Path) -> None:
    """Pipeline ingest → processed produce parquet con schema atteso."""
    raw_csv = tmp_path / "results.csv"
    raw_csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2018-09-06,France,Germany,0,0,UEFA Nations League,Munich,Germany,FALSE\n"
    )
    out_path = tmp_path / "matches.parquet"

    result_path = build_processed_matches(raw_csv, out_path)

    assert result_path == out_path
    df = pd.read_parquet(out_path)
    expected_cols = {
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
        "home_elo_before",
        "away_elo_before",
        "k_factor_used",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2
    assert df["match_id"].is_unique


def test_build_processed_matches_includes_tier1_features(tmp_path: Path) -> None:
    """matches.parquet deve includere competition_importance + days_rest_*."""
    raw_csv = tmp_path / "results.csv"
    raw_csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2018-09-06,France,Germany,0,0,UEFA Nations League,Munich,Germany,FALSE\n"
    )
    out_path = tmp_path / "matches.parquet"
    build_processed_matches(raw_csv, out_path)

    df = pd.read_parquet(out_path)
    for col in ("competition_importance", "days_rest_home", "days_rest_away", "days_rest_diff"):
        assert col in df.columns

    # France match 1 = WC → 4, match 2 = Nations League → 1
    assert df.iloc[0]["competition_importance"] == 4
    assert df.iloc[1]["competition_importance"] == 1
    # days_rest_home per Francia nella seconda riga = 53 giorni
    assert df.iloc[1]["days_rest_home"] == 53.0


def test_build_processed_matches_without_tier3_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Senza snapshots.parquet, le 6 colonne TIER3 esistono come NaN."""
    # Punta CONFIG.data_raw a un path vuoto, in modo che snapshots.parquet
    # non esista.
    fake_raw_dir = tmp_path / "fake_raw"
    fake_raw_dir.mkdir()
    monkeypatch.setattr("mondiali.data.ingestion.CONFIG.data_raw", fake_raw_dir)

    raw_csv = tmp_path / "results.csv"
    raw_csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2018-09-06,France,Germany,0,0,UEFA Nations League,Munich,Germany,FALSE\n"
    )
    out_path = tmp_path / "matches.parquet"
    build_processed_matches(raw_csv, out_path)

    df = pd.read_parquet(out_path)
    for col in TIER3_COLUMNS:
        assert col in df.columns
        assert df[col].isna().all()
