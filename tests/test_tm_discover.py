"""Test discovery degli ID reali Transfermarkt da pagine schnellsuche."""
from __future__ import annotations

from pathlib import Path

import pytest

from mondiali.data.tm_discover import parse_team_id

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture_name", "expected_slug", "expected_id"),
    [
        ("tm_search_algeria.html", "algerien", 3614),
        ("tm_search_senegal.html", "senegal", 3499),
    ],
)
def test_parse_team_id_finds_senior_via_slug_match(
    fixture_name: str, expected_slug: str, expected_id: int
) -> None:
    html = (FIXTURES / fixture_name).read_text(encoding="utf-8", errors="replace")
    assert parse_team_id(html, expected_slug) == expected_id


def test_parse_team_id_returns_none_when_slug_missing() -> None:
    html = (FIXTURES / "tm_search_algeria.html").read_text(encoding="utf-8", errors="replace")
    assert parse_team_id(html, "nonexistent-slug-xyz") is None


def test_parse_team_id_excludes_age_categories() -> None:
    """Anche se U23/U20/U17 sono presenti nei risultati, deve scegliere il senior."""
    html = (FIXTURES / "tm_search_algeria.html").read_text(encoding="utf-8", errors="replace")
    # senior = 'algerien' id 3614 (NOT 'algerien-u23' id 34867)
    assert parse_team_id(html, "algerien") == 3614


def test_parse_team_id_handles_empty_html() -> None:
    assert parse_team_id("", "italien") is None
