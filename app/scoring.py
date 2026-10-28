"""Deterministic, learned scoring for TIER CHECK.

The six visible metrics are computed by :mod:`app.analyzer`. This module turns
them into one 1–100 score with a small neural network that was trained on
labels produced by the public rubric below.

Important: the rubric is a *game rule*, not a human attractiveness study. The
model is intentionally not trained on beauty ratings or on photos of real
people. It learns a stable, explainable scoring function and nothing else.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "score_model.npz"
MODEL_META_PATH = MODEL_DIR / "score_model.json"

MODEL_VERSION = "tier-check-score-1.0.0"
RUBRIC_VERSION = "2026.10"

FEATURE_KEYS: tuple[str, ...] = ("jaw", "symmetry", "harmony", "skin", "hair", "light")
FEATURE_WEIGHTS = np.array([0.24, 0.18, 0.22, 0.16, 0.10, 0.10], dtype=np.float64)
_FEATURE_SCALE = 100.0


class ScoringError(RuntimeError):
    """The scoring model is missing or cannot be used."""


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("high must be greater than low")
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _feature_matrix(features: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a validated ``(n, 6)`` matrix clipped to the metric range."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_KEYS):
        raise ValueError(f"expected {len(FEATURE_KEYS)} metrics per row")
    return np.clip(matrix, 0.0, 100.0)


def rubric_scores(features: Sequence[float] | np.ndarray) -> np.ndarray:
    """Score metric vectors with the public TIER CHECK rubric.

    The result is the *ungated* rubric score. Tier guards are applied by
    :func:`apply_tier_guards`.
    """
    matrix = _feature_matrix(features)
    base = matrix @ FEATURE_WEIGHTS
    weakest = matrix.min(axis=1)
    consistency = 2.5 * _smoothstep(weakest, 25.0, 85.0)
    light_gate = 0.82 + 0.18 * _smoothstep(matrix[:, 5], 12.0, 60.0)
    weak_penalty = np.maximum(0.0, 30.0 - weakest) * 0.18
    scores = 7.0 + 0.90 * (base + consistency) * light_gate - weak_penalty
    return np.clip(scores, 1.0, 99.0)


def rubric_score(features: Sequence[float] | np.ndarray) -> float:
    """Ungated rubric score for one metric vector."""
    return float(rubric_scores(features)[0])


def apply_tier_guards(
    scores: Sequence[float] | np.ndarray,
    features: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Keep top tiers honest: a great average cannot hide one broken metric.

    ``CHAD`` starts at 82 and requires every metric to be at least 45.
    ``TRUE ADAM`` starts at 93 and requires every metric to be at least 65.
    """
    matrix = _feature_matrix(features)
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 0:
        values = values[None]
    if values.shape != (matrix.shape[0],):
        raise ValueError("scores and features must describe the same rows")

    weakest = matrix.min(axis=1)
    guarded = np.where(weakest < 45.0, np.minimum(values, 81.9), values)
    guarded = np.where(weakest < 65.0, np.minimum(guarded, 92.9), guarded)
    return np.clip(guarded, 1.0, 99.0)


def final_rubric_scores(features: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rubric score with tier guards applied."""
    matrix = _feature_matrix(features)
    return apply_tier_guards(rubric_scores(matrix), matrix)


def final_rubric_score(features: Sequence[float] | np.ndarray) -> float:
    """Guarded rubric score for one metric vector."""
    return float(final_rubric_scores(features)[0])


class ScoreModel:
    """Tiny NumPy MLP loaded from ``app/models/score_model.npz``."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        if not path.exists():
            raise ScoringError(
                f"Scoring model is missing: {path}. "
                "Run `python -m training.train_score_model` or reinstall the project."
            )
        with np.load(path) as data:
            self.w1 = np.asarray(data["w1"], dtype=np.float64)
            self.b1 = np.asarray(data["b1"], dtype=np.float64)
            self.w2 = np.asarray(data["w2"], dtype=np.float64)
            self.b2 = np.asarray(data["b2"], dtype=np.float64)
            self.w3 = np.asarray(data["w3"], dtype=np.float64)
            self.b3 = np.asarray(data["b3"], dtype=np.float64)
            self.feature_scale = float(data["feature_scale"])

        if self.feature_scale <= 0:
            raise ScoringError("Scoring model has an invalid feature scale.")
        if (
            self.w1.ndim != 2
            or self.w1.shape[0] != len(FEATURE_KEYS)
            or self.w1.shape[1] != self.b1.shape[0]
            or self.w2.shape[0] != self.w1.shape[1]
            or self.w2.shape[1] != self.b2.shape[0]
            or self.w3.shape[0] != self.w2.shape[1]
            or self.w3.shape[1] != 1
            or self.b3.shape != (1,)
        ):
            raise ScoringError("Scoring model has incompatible weight shapes.")

    def predict(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return raw model scores (already clamped to 1–99)."""
        matrix = _feature_matrix(features)
        hidden = np.maximum(0.0, (matrix / self.feature_scale) @ self.w1 + self.b1)
        hidden = np.maximum(0.0, hidden @ self.w2 + self.b2)
        scores = (hidden @ self.w3 + self.b3).reshape(-1) * 100.0
        return np.clip(scores, 1.0, 99.0)


_model: ScoreModel | None = None
_model_lock = threading.Lock()


def _ensure_model() -> ScoreModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ScoreModel()
    return _model


def model_scores(features: Sequence[float] | np.ndarray) -> np.ndarray:
    """Raw learned scores for one or more metric vectors."""
    return _ensure_model().predict(features)


def final_scores(features: Sequence[float] | np.ndarray) -> np.ndarray:
    """Learned scores with tier guards applied."""
    matrix = _feature_matrix(features)
    return apply_tier_guards(model_scores(matrix), matrix)


def score_from_stats(stats: Sequence[float]) -> float:
    """Score the six *displayed* integer metrics.

    Rounding here is deliberate: two scans that show the same six numbers must
    show the same score, regardless of tiny floating-point differences.
    """
    values = np.rint(np.asarray(stats, dtype=np.float64))
    if values.shape != (len(FEATURE_KEYS),):
        raise ValueError(f"expected {len(FEATURE_KEYS)} metrics")
    return round(float(final_scores(values)[0]), 1)


@lru_cache(maxsize=1)
def model_info() -> dict[str, object]:
    """Metadata written by the training script."""
    if not MODEL_META_PATH.exists():
        raise ScoringError(
            f"Scoring model metadata is missing: {MODEL_META_PATH}. "
            "Run `python -m training.train_score_model`."
        )
    payload = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScoringError("Scoring model metadata is not a JSON object.")
    return payload


def warmup() -> None:
    """Validate the artifact once during application startup."""
    _ensure_model()
    info = model_info()
    if info.get("version") != MODEL_VERSION:
        raise ScoringError(
            f"Scoring model version mismatch: {info.get('version')!r} != {MODEL_VERSION!r}."
        )
