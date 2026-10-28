"""Analyzer unit tests that do not need a real portrait."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app import analyzer, scoring
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


def test_analyze_bytes_uses_trained_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    face = np.zeros(15, dtype=np.float64)
    pointer = np.array(
        [[90.0, 100.0], [150.0, 100.0], [120.0, 130.0], [95.0, 170.0], [145.0, 170.0]],
        dtype=np.float64,
    )

    monkeypatch.setattr(analyzer, "_decode_image", lambda _raw: image)
    monkeypatch.setattr(analyzer, "_detect_largest_face", lambda _image: (face, 0.99))
    monkeypatch.setattr(
        analyzer,
        "_align_face",
        lambda _image, _face: (image, (60.0, 60.0, 120.0, 140.0), pointer, 0.0),
    )
    monkeypatch.setattr(analyzer, "_jaw_score", lambda *_args: 70.4)
    monkeypatch.setattr(analyzer, "_symmetry_score", lambda *_args: 91.4)
    monkeypatch.setattr(analyzer, "_harmony_score", lambda *_args: 96.4)
    monkeypatch.setattr(analyzer, "_skin_score", lambda *_args: 42.4)
    monkeypatch.setattr(analyzer, "_hair_score", lambda *_args: 30.4)
    monkeypatch.setattr(analyzer, "_light_score", lambda *_args: (84.4, 0.6))

    result = analyzer.analyze_bytes(b"fake-image")
    expected_stats = [70, 91, 96, 42, 30, 84]

    assert [stat.value for stat in result.stats] == expected_stats
    assert result.score == scoring.score_from_stats(expected_stats)
    assert result.score == pytest.approx(71.5, abs=0.1)
    assert result.tier == tier_for_score(result.score)
    assert result.confidence == 0.99

    repeated = analyzer.analyze_bytes(b"fake-image")
    assert repeated == result


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
