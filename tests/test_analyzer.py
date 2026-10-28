"""Analyzer unit tests that do not need a real portrait."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app import analyzer
from app.tiers import tier_for_score


def _png_bytes(
    size: tuple[int, int] = (320, 320),
    color: tuple[int, int, int] = (128, 128, 128),
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_data_url_accepts_full_url() -> None:
    raw = b"hello tier check"
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    assert analyzer.decode_data_url(data_url) == raw


def test_decode_data_url_accepts_bare_base64() -> None:
    raw = b"bare payload"
    assert analyzer.decode_data_url(base64.b64encode(raw).decode()) == raw


def test_decode_data_url_rejects_garbage() -> None:
    with pytest.raises(analyzer.InvalidImageError):
        analyzer.decode_data_url("data:image/jpeg;base64,!!!not-base64!!!")


def test_decode_data_url_rejects_non_base64_data_url() -> None:
    with pytest.raises(analyzer.InvalidImageError):
        analyzer.decode_data_url("data:image/svg+xml,<svg></svg>")


def test_analyze_rejects_non_image() -> None:
    with pytest.raises(analyzer.InvalidImageError):
        analyzer.analyze_bytes(b"<!doctype html><html><body>nope</body></html>")


def test_analyze_rejects_empty_payload() -> None:
    with pytest.raises(analyzer.InvalidImageError):
        analyzer.analyze_bytes(b"")


def test_analyze_rejects_oversized_payload() -> None:
    with pytest.raises(analyzer.InvalidImageError):
        analyzer.analyze_bytes(b"x" * (analyzer.MAX_IMAGE_BYTES + 1))


def test_analyze_finds_no_face_in_flat_image() -> None:
    with pytest.raises(analyzer.FaceNotFoundError):
        analyzer.analyze_bytes(_png_bytes())


def test_bell_and_clamp() -> None:
    assert analyzer._bell(0.5, 0.5, 0.2) == 1.0
    assert analyzer._bell(0.7, 0.5, 0.2) == pytest.approx(0.0, abs=1e-9)
    assert analyzer._bell(0.6, 0.5, 0.2) == pytest.approx(0.75)
    assert analyzer._clamp01(-4) == 0.0
    assert analyzer._clamp01(0.42) == 0.42
    assert analyzer._clamp01(9) == 1.0


def test_compose_score_is_deterministic_and_bounded() -> None:
    args = (61.0, 74.0, 68.0, 52.0, 48.0, 88.0, 123456)
    first = analyzer._compose_score(*args)
    second = analyzer._compose_score(*args)
    assert first == second
    assert 1.0 <= first <= 99.4


def test_compose_score_seed_changes_score() -> None:
    args = (61.0, 74.0, 68.0, 52.0, 48.0, 88.0)
    scores = {analyzer._compose_score(*args, seed) for seed in range(1, 25)}
    assert len(scores) > 1


def test_notes_are_short_and_strings() -> None:
    tier = tier_for_score(62.0)
    stats = (
        analyzer.Stat("jaw", "ЧЕЛЮСТЬ", 40),
        analyzer.Stat("symmetry", "СИММЕТРИЯ", 41),
        analyzer.Stat("harmony", "ПРОПОРЦИИ", 42),
        analyzer.Stat("skin", "КОЖА", 43),
        analyzer.Stat("hair", "ОБЪЁМ", 44),
        analyzer.Stat("light", "СВЕТ", 45),
    )
    notes = analyzer._notes_for(tier, stats, 0.2, 0.5)
    assert 1 <= len(notes) <= 2
    assert all(isinstance(note, str) and note for note in notes)


def test_other_notes_are_used_for_good_scans() -> None:
    tier = tier_for_score(90.0)
    stats = (
        analyzer.Stat("jaw", "ЧЕЛЮСТЬ", 95),
        analyzer.Stat("symmetry", "СИММЕТРИЯ", 95),
        analyzer.Stat("harmony", "ПРОПОРЦИИ", 95),
        analyzer.Stat("skin", "КОЖА", 95),
        analyzer.Stat("hair", "ОБЪЁМ", 95),
        analyzer.Stat("light", "СВЕТ", 95),
    )
    notes = analyzer._notes_for(tier, stats, 0.2, 0.55)
    assert len(notes) == 2
    assert "мем" in notes[-1].lower()
