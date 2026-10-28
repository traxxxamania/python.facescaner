"""Tier ladder contract tests."""

from __future__ import annotations

import pytest

from app.tiers import TIERS, tier_for_score


def test_exactly_seven_tiers() -> None:
    assert len(TIERS) == 7


def test_tier_names_and_order() -> None:
    assert [tier.name for tier in TIERS] == [
        "SUB 3",
        "SUB 5",
        "LTN",
        "MTN",
        "HTN",
        "CHAD",
        "TRUE ADAM",
    ]


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "sub3"),
        (24.9, "sub3"),
        (25.0, "sub5"),
        (39.9, "sub5"),
        (40.0, "ltn"),
        (54.9, "ltn"),
        (55.0, "mtn"),
        (69.9, "mtn"),
        (70.0, "htn"),
        (81.9, "htn"),
        (82.0, "chad"),
        (92.9, "chad"),
        (93.0, "true_adam"),
        (100.0, "true_adam"),
        (-20, "sub3"),
        (140, "true_adam"),
    ],
)
def test_tier_boundaries(score: float, expected: str) -> None:
    assert tier_for_score(score).id == expected


def test_every_tier_has_display_data() -> None:
    for tier in TIERS:
        assert tier.name
        assert tier.tag
        assert tier.description
        assert tier.color.startswith("#")
        assert len(tier.color) == 7
        assert 0 < tier.max_score <= 100
        assert tier.rarity
        assert tier.emoji
