"""Train the small TIER CHECK score model.

Run from the repository root:

    python -m training.train_score_model

The training set is generated from the public rubric in :mod:`app.scoring`.
No photos, human ratings or external datasets are used. The script is fully
deterministic for a fixed seed and writes both the weights and a model card.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.scoring import (  # noqa: E402
    FEATURE_KEYS,
    MODEL_META_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    RUBRIC_VERSION,
    apply_tier_guards,
    final_rubric_scores,
    rubric_scores,
)
from app.tiers import TIERS, tier_for_score  # noqa: E402

SEED = 20261028
HIDDEN = (32, 24)
ANCHORS: dict[str, list[int]] = {
    "demo_portrait": [70, 91, 96, 42, 30, 84],
    "balanced_70": [70, 70, 70, 70, 70, 70],
    "weak_20": [20, 20, 20, 20, 20, 20],
    "elite_95": [95, 95, 95, 95, 95, 95],
    "one_weak_metric": [95, 95, 95, 95, 95, 40],
}


def make_dataset(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample metric vectors with a mix of realistic and edge-case coverage."""
    if size < 4000:
        raise ValueError("use at least 4000 samples")

    quality = rng.normal(62.0, 15.0, size)
    features = 0.58 * quality[:, None] + 0.42 * rng.normal(55.0, 20.0, (size, 6))
    features = np.rint(np.clip(features, 0.0, 100.0))

    # Uniform coverage makes the model reliable at the edges of the rubric.
    uniform = size // 5
    features[:uniform] = rng.integers(0, 101, size=(uniform, 6))

    # Targeted cases keep rare tiers and one-weak-metric combinations represented.
    targeted = min(10_000, size // 10)
    for index in range(uniform, uniform + targeted):
        kind = index % 4
        if kind == 0:
            features[index] = rng.integers(0, 35, size=6)
        elif kind == 1:
            features[index] = rng.integers(75, 101, size=6)
        elif kind == 2:
            features[index] = rng.integers(60, 96, size=6)
            features[index, index % 6] = rng.integers(0, 45)
        else:
            features[index] = rng.integers(85, 101, size=6)
            features[index, index % 6] = rng.integers(45, 70)

    return features.astype(np.float32)


def init_weights(rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    shapes = ((len(FEATURE_KEYS), HIDDEN[0]), (HIDDEN[0], HIDDEN[1]), (HIDDEN[1], 1))
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for inputs, outputs in shapes:
        weights.append(
            rng.normal(0.0, np.sqrt(2.0 / inputs), (inputs, outputs)).astype(np.float32)
        )
        biases.append(np.zeros(outputs, dtype=np.float32))
    return weights, biases


def forward(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    features: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    activations = [features]
    hidden = features
    for index in range(len(weights) - 1):
        hidden = np.maximum(0.0, hidden @ weights[index] + biases[index])
        activations.append(hidden)
    output = (hidden @ weights[-1] + biases[-1]).reshape(-1)
    return output, activations


def predict(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    output, _ = forward(weights, biases, features)
    return np.clip(output * 100.0, 1.0, 99.0)


def train(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    rng = np.random.default_rng(seed + 1)
    weights, biases = init_weights(rng)
    moment_w = [np.zeros_like(weight) for weight in weights]
    velocity_w = [np.zeros_like(weight) for weight in weights]
    moment_b = [np.zeros_like(bias) for bias in biases]
    velocity_b = [np.zeros_like(bias) for bias in biases]

    best_mae = float("inf")
    best_weights = [weight.copy() for weight in weights]
    best_biases = [bias.copy() for bias in biases]
    step = 0
    started = time.perf_counter()

    for epoch in range(epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(train_x), batch_size):
            batch = order[start : start + batch_size]
            batch_x = train_x[batch]
            batch_y = train_y[batch]
            output, activations = forward(weights, biases, batch_x)
            error = output - batch_y
            gradient = (2.0 / len(batch) * error)[:, None].astype(np.float32)

            grad_w: list[np.ndarray] = []
            grad_b: list[np.ndarray] = []
            for index in range(len(weights) - 1, -1, -1):
                grad_w.insert(0, activations[index].T @ gradient)
                grad_b.insert(0, gradient.sum(axis=0))
                if index > 0:
                    gradient = (gradient @ weights[index].T) * (activations[index] > 0)

            step += 1
            for index in range(len(weights)):
                moment_w[index] = 0.9 * moment_w[index] + 0.1 * grad_w[index]
                velocity_w[index] = (
                    0.999 * velocity_w[index] + 0.001 * grad_w[index] ** 2
                )
                moment_b[index] = 0.9 * moment_b[index] + 0.1 * grad_b[index]
                velocity_b[index] = (
                    0.999 * velocity_b[index] + 0.001 * grad_b[index] ** 2
                )
                corrected_w = moment_w[index] / (1.0 - 0.9**step)
                corrected_b = moment_b[index] / (1.0 - 0.9**step)
                scale_w = np.sqrt(velocity_w[index] / (1.0 - 0.999**step)) + 1e-8
                scale_b = np.sqrt(velocity_b[index] / (1.0 - 0.999**step)) + 1e-8
                weights[index] -= learning_rate * corrected_w / scale_w
                biases[index] -= learning_rate * corrected_b / scale_b

        validation, _ = forward(weights, biases, val_x)
        validation_mae = float(np.mean(np.abs(validation - val_y)) * 100.0)
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_weights = [weight.copy() for weight in weights]
            best_biases = [bias.copy() for bias in biases]

        if epoch % 20 == 0 or epoch == epochs - 1:
            elapsed = time.perf_counter() - started
            print(
                f"epoch {epoch:3d}/{epochs}  val MAE {validation_mae:5.3f}  "
                f"best {best_mae:5.3f}  {elapsed:4.1f}s",
                flush=True,
            )

    return best_weights, best_biases, best_mae


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - residual / total if total else 1.0


def tier_distribution(scores: np.ndarray) -> dict[str, dict[str, float | int]]:
    counts = {tier.id: 0 for tier in TIERS}
    for score in scores:
        counts[tier_for_score(float(score)).id] += 1
    total = max(1, len(scores))
    return {
        tier.id: {
            "name": tier.name,
            "count": counts[tier.id],
            "share": round(counts[tier.id] / total, 4),
        }
        for tier in TIERS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TIER CHECK score model.")
    parser.add_argument("--samples", type=int, default=80_000)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.samples < 4000:
        parser.error("--samples must be at least 4000")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    rng = np.random.default_rng(args.seed)
    features = make_dataset(args.samples, rng)
    labels = (rubric_scores(features) / 100.0).astype(np.float32)
    scaled = (features / 100.0).astype(np.float32)

    split_rng = np.random.default_rng(args.seed + 17)
    order = split_rng.permutation(len(features))
    train_end = int(len(features) * 0.75)
    val_end = int(len(features) * 0.875)
    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    train_x, train_y = scaled[train_idx], labels[train_idx]
    val_x, val_y = scaled[val_idx], labels[val_idx]
    test_x = scaled[test_idx]
    test_x_raw = features[test_idx]

    print(
        f"dataset {len(features):,} | train {len(train_x):,} | "
        f"val {len(val_x):,} | test {len(test_x):,}",
        flush=True,
    )
    best_weights, best_biases, best_mae = train(
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=2e-3,
        seed=args.seed,
    )

    raw_test = predict(best_weights, best_biases, test_x)
    final_test = apply_tier_guards(raw_test, test_x_raw)
    rubric_raw = rubric_scores(test_x_raw)
    rubric_final = final_rubric_scores(test_x_raw)

    # Practical monotonicity check: bump a random metric and look for drops.
    check_rng = np.random.default_rng(args.seed + 99)
    base = check_rng.integers(0, 101, size=(10_000, len(FEATURE_KEYS))).astype(np.float32)
    bumped = base.copy()
    dimension = check_rng.integers(0, len(FEATURE_KEYS), size=len(base))
    step = check_rng.integers(1, 21, size=len(base))
    rows = np.arange(len(base))
    bumped[rows, dimension] = np.minimum(100, bumped[rows, dimension] + step)
    base_scores = predict(best_weights, best_biases, base / 100.0)
    bumped_scores = predict(best_weights, best_biases, bumped / 100.0)
    violations = np.maximum(0.0, base_scores - bumped_scores)

    anchor_features = np.array(list(ANCHORS.values()), dtype=np.float32)
    anchor_rubric = final_rubric_scores(anchor_features)
    anchor_model = apply_tier_guards(
        predict(best_weights, best_biases, anchor_features / 100.0),
        anchor_features,
    )

    metrics = {
        "val_mae": round(best_mae, 4),
        "test_mae_raw": round(float(np.mean(np.abs(raw_test - rubric_raw))), 4),
        "test_mae_final": round(float(np.mean(np.abs(final_test - rubric_final))), 4),
        "test_max_abs_error": round(float(np.max(np.abs(final_test - rubric_final))), 4),
        "test_r2_final": round(r2_score(rubric_final, final_test), 6),
        "monotonicity_violations": int(np.sum(violations > 1e-6)),
        "monotonicity_max_drop": round(float(violations.max()), 4),
    }
    info = {
        "name": "TIER CHECK learned score",
        "version": MODEL_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "created": date.today().isoformat(),
        "features": list(FEATURE_KEYS),
        "architecture": [len(FEATURE_KEYS), *HIDDEN, 1],
        "activation": "relu",
        "training": {
            "samples": len(features),
            "train": len(train_x),
            "validation": len(val_x),
            "test": len(test_x),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 2e-3,
            "seed": args.seed,
        },
        "metrics": metrics,
        "tier_distribution": tier_distribution(final_test),
        "anchors": {
            name: {
                "stats": values,
                "rubric": round(float(anchor_rubric[index]), 1),
                "model": round(float(anchor_model[index]), 1),
            }
            for index, (name, values) in enumerate(ANCHORS.items())
        },
        "disclaimer": (
            "Trained on a self-defined rubric and synthetic metric vectors. "
            "Not trained on human attractiveness ratings or personal photos."
        ),
    }

    MODEL_DIR = MODEL_PATH.parent
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        w1=best_weights[0],
        b1=best_biases[0],
        w2=best_weights[1],
        b2=best_biases[1],
        w3=best_weights[2],
        b3=best_biases[2],
        feature_scale=np.array(100.0, dtype=np.float32),
    )
    metadata = json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    MODEL_META_PATH.write_text(metadata, encoding="utf-8")
    (Path(__file__).resolve().parent / "report.json").write_text(metadata, encoding="utf-8")

    print(
        f"test MAE: {metrics['test_mae_final']:.3f} | "
        f"max error: {metrics['test_max_abs_error']:.3f} | "
        f"R2: {metrics['test_r2_final']:.4f}",
        flush=True,
    )
    print(
        f"monotonicity: {metrics['monotonicity_violations']} violations, "
        f"max drop {metrics['monotonicity_max_drop']:.2f}",
        flush=True,
    )
    for name, anchor in info["anchors"].items():
        print(f"anchor {name:16s} rubric {anchor['rubric']:5.1f}  model {anchor['model']:5.1f}")
    print(f"saved {MODEL_PATH}", flush=True)
    print(f"saved {MODEL_META_PATH}", flush=True)


if __name__ == "__main__":
    main()
