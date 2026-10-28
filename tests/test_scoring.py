"""Tests for the deterministic, learned TIER CHECK scoring system."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app import scoring


def test_rubric_is_monotonic_for_each_metric() -> None:
    rng = np.random.default_rng(20261028)
    base = rng.integers(0, 100, size=(2000, len(scoring.FEATURE_KEYS))).astype(np.float64)
    before = scoring.rubric_scores(base)

    for dimension in range(len(scoring.FEATURE_KEYS)):
        bumped = base.copy()
        bumped[:, dimension] = np.minimum(100, bumped[:, dimension] + 1)
        after = scoring.rubric_scores(bumped)
        assert np.all(after >= before - 1e-9)


def test_rubric_scores_are_bounded() -> None:
    edges = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [100, 100, 100, 100, 100, 100],
            [70, 91, 96, 42, 30, 84],
        ],
        dtype=np.float64,
    )
    scores = scoring.rubric_scores(edges)
    assert np.all(scores >= 1.0)
    assert np.all(scores <= 99.0)
    assert scoring.rubric_score([70, 91, 96, 42, 30, 84]) == pytest.approx(72.2, abs=0.05)


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ([70, 91, 96, 42, 30, 84], 72.2),
        ([70, 70, 70, 70, 70, 70], 71.9),
        ([20, 20, 20, 20, 20, 20], 20.2),
        ([95, 95, 95, 95, 95, 95], 94.8),
    ],
)
def test_rubric_anchors(stats: list[int], expected: float) -> None:
    assert scoring.final_rubric_score(stats) == pytest.approx(expected, abs=0.1)


def test_tier_guards_cap_one_weak_metric() -> None:
    weak_chad = np.array([[95, 95, 95, 95, 95, 40]], dtype=np.float64)
    guarded = scoring.apply_tier_guards([99.0], weak_chad)
    assert guarded[0] == pytest.approx(81.9)

    weak_adam = np.array([[95, 95, 95, 95, 95, 60]], dtype=np.float64)
    guarded = scoring.apply_tier_guards([99.0], weak_adam)
    assert guarded[0] == pytest.approx(92.9)

    clean = np.array([[95, 95, 95, 95, 95, 95]], dtype=np.float64)
    guarded = scoring.apply_tier_guards([94.8], clean)
    assert guarded[0] == pytest.approx(94.8)


def test_model_tracks_rubric_on_fresh_samples() -> None:
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 101, size=(2000, len(scoring.FEATURE_KEYS))).astype(np.float64)
    error = np.abs(scoring.final_scores(raw) - scoring.final_rubric_scores(raw))
    assert float(error.mean()) < 0.7
    assert float(error.max()) < 6.0


def test_model_practically_monotonic() -> None:
    rng = np.random.default_rng(99)
    base = rng.integers(0, 100, size=(2000, len(scoring.FEATURE_KEYS))).astype(np.float64)
    bumped = base.copy()
    dimension = rng.integers(0, len(scoring.FEATURE_KEYS), size=len(base))
    rows = np.arange(len(base))
    bumped[rows, dimension] = np.minimum(100, bumped[rows, dimension] + 1)

    drop = scoring.model_scores(base) - scoring.model_scores(bumped)
    assert float(drop.max()) < 1.0


def test_score_from_stats_is_deterministic_and_rounded() -> None:
    stats = [70, 91, 96, 42, 30, 84]
    first = scoring.score_from_stats(stats)
    second = scoring.score_from_stats(stats)
    fractional = scoring.score_from_stats([70.4, 91.4, 96.4, 42.4, 30.4, 84.4])

    assert first == second == fractional
    assert first == pytest.approx(71.5, abs=0.1)
    assert 1.0 <= first <= 99.0


def test_score_depends_on_stats_not_input_order_type() -> None:
    stats = [70, 91, 96, 42, 30, 84]
    assert scoring.score_from_stats(stats) == scoring.score_from_stats(tuple(stats))
    assert scoring.score_from_stats(stats) != scoring.score_from_stats([95, 95, 95, 95, 95, 95])


def test_model_metadata_matches_artifact() -> None:
    info = scoring.model_info()
    assert info["version"] == scoring.MODEL_VERSION
    assert info["rubric_version"] == scoring.RUBRIC_VERSION
    assert info["features"] == list(scoring.FEATURE_KEYS)
    assert info["architecture"] == [len(scoring.FEATURE_KEYS), 32, 24, 1]

    metrics = info["metrics"]
    assert metrics["test_mae_final"] < 1.0
    assert metrics["test_max_abs_error"] < 5.0
    assert metrics["test_r2_final"] > 0.99
    assert metrics["monotonicity_max_drop"] < 1.0

    anchors = info["anchors"]
    assert anchors["demo_portrait"]["stats"] == [70, 91, 96, 42, 30, 84]
    for anchor in anchors.values():
        assert abs(anchor["model"] - anchor["rubric"]) <= 1.0


def test_model_info_is_cached_json() -> None:
    info = scoring.model_info()
    assert json.dumps(info)  # JSON-serialisable payload
    assert scoring.model_info() is info


def test_missing_model_raises_helpful_error(tmp_path) -> None:
    with pytest.raises(scoring.ScoringError, match="Scoring model is missing"):
        scoring.ScoreModel(tmp_path / "missing.npz")
