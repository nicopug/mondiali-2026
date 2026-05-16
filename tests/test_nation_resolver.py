"""Tests for nation_resolver.py."""
from __future__ import annotations

import pytest

from mondiali.inference.nation_resolver import NationNotFound, NationResolver


@pytest.fixture
def resolver() -> NationResolver:
    return NationResolver(canonical_names=[
        "United States", "England", "France", "Italy", "Spain",
        "Germany", "Brazil", "Argentina", "Korea Republic",
        "Côte d'Ivoire", "Bosnia and Herzegovina",
    ])


def test_exact_match(resolver: NationResolver) -> None:
    assert resolver.resolve("France") == "France"


def test_case_insensitive_match(resolver: NationResolver) -> None:
    assert resolver.resolve("france") == "France"
    assert resolver.resolve("FRANCE") == "France"


def test_alias_usa(resolver: NationResolver) -> None:
    assert resolver.resolve("USA") == "United States"
    assert resolver.resolve("usa") == "United States"
    assert resolver.resolve("US") == "United States"


def test_alias_south_korea(resolver: NationResolver) -> None:
    assert resolver.resolve("South Korea") == "Korea Republic"


def test_typo_suggestion(resolver: NationResolver) -> None:
    with pytest.raises(NationNotFound) as exc_info:
        resolver.resolve("Frnace")  # typo
    assert "France" in exc_info.value.suggestions


def test_unknown_no_suggestion(resolver: NationResolver) -> None:
    with pytest.raises(NationNotFound) as exc_info:
        resolver.resolve("Atlantis")
    # Suggestions list could be empty
    assert exc_info.value.query == "Atlantis"


def test_empty_string_raises(resolver: NationResolver) -> None:
    with pytest.raises(NationNotFound):
        resolver.resolve("")
    with pytest.raises(NationNotFound):
        resolver.resolve("   ")


def test_error_message_contains_suggestions(resolver: NationResolver) -> None:
    try:
        resolver.resolve("Englan")  # missing d
    except NationNotFound as e:
        assert "England" in str(e)
        assert "Did you mean" in str(e)
