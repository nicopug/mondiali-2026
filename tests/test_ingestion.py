"""Test per data ingestion: download e parsing di international_results."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mondiali.data.ingestion import (
    INTERNATIONAL_RESULTS_URL,
    download_international_results,
    load_international_results,
)


def test_download_writes_csv_to_destination(tmp_path: Path) -> None:
    """Il download scrive il CSV alla destinazione specificata."""
    dest = tmp_path / "results.csv"
    fake_csv = b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE\n"

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
