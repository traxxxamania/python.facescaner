"""Heuristic face-tier analyzer.

The goal is a fast, deterministic, *playful* score that runs on a laptop CPU.
It is not a beauty model, a medical tool, or a judgement of a real person.
Everything is processed in memory and never written to disk.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .tiers import Tier, tier_for_score

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2026may.onnx"
MAX_IMAGE_BYTES = 8_000_000
MAX_PIXELS = 24_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

_DETECTOR_CONFIDENCE = 0.62
_DETECTOR_NMS = 0.3
_DETECTOR_TOP_K = 500
_MAX_SIDE = 1600


class AnalyzerError(Exception):
    """Base class for user-facing analyzer errors."""


class InvalidImageError(AnalyzerError):
    """The uploaded payload is not a supported image."""


class FaceNotFoundError(AnalyzerError):
    """No face was found in the image."""


@dataclass(frozen=True, slots=True)
class Stat:
    key: str
    label: str
    value: int


@dataclass(frozen=True, slots=True)
class Analysis:
    score: float
    tier: Tier
    stats: tuple[Stat, ...]
    notes: tuple[str, ...]
    face_box: tuple[float, float, float, float]
    confidence: float


_detector: cv2.FaceDetectorYN | None = None
_detector_lock = threading.Lock()


def warmup() -> None:
    """Load the ONNX detector once so the first request stays snappy."""
    _ensure_detector(640, 640)


def _ensure_detector(width: int, height: int) -> cv2.FaceDetectorYN:
    global _detector
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"YuNet model is missing: {MODEL_PATH}. "
            "Download it from opencv_zoo or reinstall the project."
        )
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (width, height),
            _DETECTOR_CONFIDENCE,
            _DETECTOR_NMS,
            _DETECTOR_TOP_K,
        )
    else:
        _detector.setInputSize((width, height))
    return _detector


def _detect_largest_face(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    with _detector_lock:
        detector = _ensure_detector(width, height)
        _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        raise FaceNotFoundError(
            "Лицо не найдено. Встань ровно, сними маску и посмотри в камеру — "
            "или загрузи фото, где лицо видно целиком."
        )

    areas = faces[:, 2] * faces[:, 3] * faces[:, 14]
    face = faces[int(np.argmax(areas))]
    return face.astype(np.float64), float(face[14])


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _bell(value: float, center: float, tolerance: float, power: float = 2.0) -> float:
    """A smooth 1.0-at-center score that reaches 0 at ±tolerance."""
    distance = abs(value - center) / tolerance
    return float(max(0.0, 1.0 - distance**power))


def _align_face(
    image: np.ndarray, face: np.ndarray
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray, float]:
    """Rotate the frame so the eye line is horizontal."""
    height, width = image.shape[:2]
    x, y, fw, fh = (float(value) for value in face[:4])
    right_eye_x, right_eye_y, left_eye_x, left_eye_y = (
        float(value) for value in face[4:8]
    )

    angle = math.atan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x)
    angle = max(-0.5, min(0.5, angle))
    cos, sin = math.cos(angle), math.sin(angle)
    center_x = (right_eye_x + left_eye_x) / 2
    center_y = (right_eye_y + left_eye_y) / 2

    rotation = np.array(
        [
            [cos, sin, center_x - cos * center_x - sin * center_y],
            [-sin, cos, center_y + sin * center_x - cos * center_y],
        ],
        dtype=np.float32,
    )
    rotated = cv2.warpAffine(
        image,
        rotation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    points = np.array(
        [
            [right_eye_x, right_eye_y],
            [left_eye_x, left_eye_y],
            *[
                [float(value) for value in face[8 + index * 2 : 10 + index * 2]]
                for index in range(3)
            ],
        ],
        dtype=np.float64,
    )
    points = points @ rotation[:, :2].T + rotation[:, 2]

    corners = np.array(
        [[x, y], [x + fw, y], [x, y + fh], [x + fw, y + fh]], dtype=np.float64
    )
    corners = corners @ rotation[:, :2].T + rotation[:, 2]
    min_x, min_y = corners.min(axis=0)
    max_x, max_y = corners.max(axis=0)
    return rotated, (min_x, min_y, max_x - min_x, max_y - min_y), points, abs(angle)


def _symmetry_score(points: np.ndarray) -> float:
    (right_eye_x, right_eye_y), (left_eye_x, left_eye_y), (nose_x, _), (
        right_mouth_x,
        right_mouth_y,
    ), (left_mouth_x, left_mouth_y) = points

    eye_mid_x = (right_eye_x + left_eye_x) / 2
    eye_distance = max(1e-6, math.hypot(left_eye_x - right_eye_x, left_eye_y - right_eye_y))
    mouth_mid_x = (right_mouth_x + left_mouth_x) / 2

    nose_offset = abs(nose_x - eye_mid_x) / eye_distance
    mouth_offset = abs(mouth_mid_x - eye_mid_x) / eye_distance
    mouth_level = abs(right_mouth_y - left_mouth_y) / eye_distance

    return (
        _bell(nose_offset, 0.0, 0.30, 1.4)
        + _bell(mouth_offset, 0.0, 0.30, 1.4)
        + _bell(mouth_level, 0.0, 0.20, 1.4)
    ) / 3 * 100


def _harmony_score(
    points: np.ndarray,
    x: float,
    y: float,
    fw: float,
    fh: float,
    eye_distance: float,
) -> float:
    (_, right_eye_y), (_, left_eye_y), (_, nose_y), (
        right_mouth_x,
        right_mouth_y,
    ), (left_mouth_x, left_mouth_y) = points

    eye_mid_y = (right_eye_y + left_eye_y) / 2
    mouth_mid_y = (right_mouth_y + left_mouth_y) / 2
    mouth_width = math.hypot(left_mouth_x - right_mouth_x, left_mouth_y - right_mouth_y)

    parts = (
        _bell(fw / fh, 0.75, 0.12, 1.6),
        _bell((eye_mid_y - y) / fh, 0.40, 0.11, 1.6),
        _bell(eye_distance / fw, 0.46, 0.10, 1.6),
        _bell(
            (nose_y - eye_mid_y) / max(1e-6, mouth_mid_y - eye_mid_y),
            0.58,
            0.15,
            1.6,
        ),
        _bell(mouth_width / fw, 0.40, 0.11, 1.6),
    )
    return sum(parts) / len(parts) * 100


def _jaw_score(
    rotated: np.ndarray,
    box: tuple[float, float, float, float],
    points: np.ndarray,
) -> float:
    x, y, fw, fh = box
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.GaussianBlur(cv2.magnitude(grad_x, grad_y), (3, 3), 0)
    height, width = gray.shape

    # A quadratic arc that approximates the lower jaw inside the face box.
    p0 = np.array([x + 0.02 * fw, y + 0.58 * fh])
    p1 = np.array([x + 0.50 * fw, y + 1.04 * fh])
    p2 = np.array([x + 0.98 * fw, y + 0.58 * fh])
    samples: list[float] = []
    for index in range(21):
        t = 0.05 + 0.90 * index / 20
        point_x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        point_y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        px, py = round(point_x), round(point_y)
        if 2 <= px < width - 2 and 2 <= py < height - 2:
            samples.append(float(magnitude[py - 1 : py + 2, px - 1 : px + 2].max()))

    arc_strength = float(np.mean(samples)) if samples else 0.0
    lower_y0 = max(0, int(y + 0.55 * fh))
    lower_y1 = min(height, int(y + fh))
    lower_x0 = max(0, int(x + 0.10 * fw))
    lower_x1 = min(width, int(x + 0.90 * fw))
    lower_region = magnitude[lower_y0:lower_y1, lower_x0:lower_x1]
    lower_mean = float(lower_region.mean()) if lower_region.size else 1.0
    arc_ratio = arc_strength / (lower_mean + 1e-6)

    (_, _), (_, _), (_, _), (right_mouth_x, right_mouth_y), (
        left_mouth_x,
        left_mouth_y,
    ) = points
    mouth_width = math.hypot(left_mouth_x - right_mouth_x, left_mouth_y - right_mouth_y)
    mouth_mid_y = (right_mouth_y + left_mouth_y) / 2
    lower_third = (y + fh - mouth_mid_y) / fh

    score = (
        _bell(arc_ratio, 1.45, 0.70, 1.5) * 0.35
        + _bell(lower_third, 0.30, 0.11, 1.5) * 0.25
        + _bell(mouth_width / fw, 0.40, 0.11, 1.5) * 0.20
        + _bell(fw / fh, 0.76, 0.11, 1.5) * 0.20
    )
    return score * 100


def _skin_score(crop: np.ndarray) -> float:
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    y0, y1 = int(height * 0.38), int(height * 0.68)
    left_x0, left_x1 = int(width * 0.10), int(width * 0.36)
    right_x0, right_x1 = int(width * 0.64), int(width * 0.90)

    cheeks = [
        gray[y0:y1, left_x0:left_x1],
        gray[y0:y1, right_x0:right_x1],
    ]
    if any(cheek.size == 0 for cheek in cheeks):
        return 50.0

    texture = float(
        np.mean(
            [
                np.mean(
                    np.abs(
                        cheek.astype(np.float32)
                        - cv2.GaussianBlur(cheek.astype(np.float32), (0, 0), 2.5)
                    )
                )
                for cheek in cheeks
            ]
        )
    )
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 12)
    shade = float(
        np.mean(
            [
                float(np.std(smooth[y0:y1, left_x0:left_x1])),
                float(np.std(smooth[y0:y1, right_x0:right_x1])),
            ]
        )
    )
    return (
        _clamp01(np.exp(-texture / 6.0)) * 0.65
        + _clamp01(np.exp(-shade / 22.0)) * 0.35
    ) * 100


def _hair_score(
    rotated: np.ndarray,
    x: float,
    y: float,
    fw: float,
    fh: float,
) -> float:
    _, width = rotated.shape[:2]
    y0 = max(0, int(y - 0.50 * fh))
    y1 = max(0, int(y + 0.08 * fh))
    x0 = max(0, int(x - 0.20 * fw))
    x1 = min(width, int(x + 1.20 * fw))
    region = cv2.cvtColor(rotated[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if region.size < 100:
        return 50.0

    region = cv2.resize(region, (160, 80), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(region, 40, 120)
    edge_density = float(np.mean(edges > 0))
    dark_ratio = float(np.mean(region < np.percentile(region, 35)))
    texture_std = float(region.std())

    score = (
        _clamp01((edge_density - 0.015) / 0.20) * 0.45
        + _clamp01((dark_ratio - 0.2) / 0.6) * 0.15
        + _clamp01((texture_std - 12) / 45) * 0.40
    )
    return score * 100


def _light_score(crop: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()) / 255
    contrast = float(gray.std()) / 64
    overexposed = float(np.mean(gray > 247))
    underexposed = float(np.mean(gray < 10))
    score = (
        _bell(brightness, 0.55, 0.35)
        + _bell(contrast, 0.9, 0.8)
        + _clamp01(1 - (overexposed + underexposed) / 0.30)
    ) / 3 * 100
    return score, brightness


def _compose_score(
    symmetry: float,
    jaw: float,
    harmony: float,
    skin: float,
    hair: float,
    light: float,
    seed: int,
) -> float:
    raw = (
        symmetry * 0.18
        + jaw * 0.24
        + harmony * 0.22
        + skin * 0.16
        + hair * 0.10
        + light * 0.10
    )
    # A stable ±1.5-point jitter keeps repeated scans interesting without
    # turning the same photo into a different tier every time.
    jitter = ((seed % 2001) / 2000 - 0.5) * 3.0
    return float(max(1.0, min(99.4, 13.0 + raw * 0.85 + jitter)))


def _notes_for(
    tier: Tier,
    stats: tuple[Stat, ...],
    face_area_ratio: float,
    brightness: float,
) -> tuple[str, ...]:
    values = {stat.key: stat.value for stat in stats}
    notes: list[str] = []

    if face_area_ratio < 0.030:
        notes.append("Лицо слишком далеко. Подойди ближе — камере нужен не пейзаж, а ты.")
    elif face_area_ratio > 0.65:
        notes.append("Слишком близко. Отойди на полшага, иначе в кадр влезет только нос.")

    if brightness < 0.24:
        notes.append("Темновато. Развернись к окну: свет — самый бесплатный апгрейд.")

    tips = (
        (
            "symmetry",
            60,
            "Симметрия просела: держи голову ровнее и не наклоняй телефон.",
        ),
        (
            "jaw",
            55,
            "Челюсть потерялась в тени. Чуть выдвини подбородок и добавь боковой свет.",
        ),
        (
            "harmony",
            60,
            "Пропорции съел ракурс. Подними телефон на уровень глаз.",
        ),
        (
            "skin",
            50,
            "Кожа: камера видит всё. Свет и сон работают лучше любого фильтра.",
        ),
        (
            "hair",
            45,
            "Объём просел. Распуши волосы или надень что-то с характером.",
        ),
        (
            "light",
            55,
            "Света маловато. Окно > кольцевая лампа > свет из холодильника.",
        ),
    )

    weak = sorted(
        (item for item in tips if values[item[0]] < item[1]),
        key=lambda item: values[item[0]],
    )
    for _key, _, tip in weak:
        if len(notes) >= 2:
            break
        if tip not in notes:
            notes.append(tip)

    if not notes:
        if tier.id in {"chad", "true_adam"}:
            notes.append("Алгоритм доволен. Не забудь выключить камеру перед сном.")
        elif tier.id == "htn":
            notes.append("Этот кадр можно ставить на аву. Даже без фильтра.")
        else:
            notes.append("Норм кадр. Свет и ракурс — и алгоритм скажет больше.")

    if len(notes) < 2:
        notes.append("Это мем, а не диагноз. Улыбнись и живи свою жизнь.")

    return tuple(notes[:2])


def _decode_image(raw: bytes) -> np.ndarray:
    if not raw:
        raise InvalidImageError("Пустой файл. Попробуй другое фото.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise InvalidImageError("Файл больше 8 МБ. Сожми фото и попробуй снова.")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise InvalidImageError("Поддерживаются JPG, PNG и WEBP.")
            image = ImageOps.exif_transpose(image)
            image.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.LANCZOS)
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            "Не удалось прочитать изображение. Попробуй JPG или PNG."
        ) from exc

    if array.size == 0:
        raise InvalidImageError("Пустое изображение. Попробуй другое фото.")

    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def decode_data_url(data_url: str) -> bytes:
    """Accept both a full data URL and a bare base64 string."""
    payload = data_url.strip()
    if payload.startswith("data:"):
        try:
            header, payload = payload.split(",", 1)
        except ValueError as exc:
            raise InvalidImageError("Некорректный data URL.") from exc
        if ";base64" not in header.lower():
            raise InvalidImageError("Ожидалось base64-изображение.")

    payload = "".join(payload.split())
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("Некорректная base64-строка.") from exc


def analyze_bytes(raw: bytes) -> Analysis:
    image = _decode_image(raw)
    height, width = image.shape[:2]
    face, confidence = _detect_largest_face(image)

    seed = int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "big")
    x, y, fw, fh = (float(value) for value in face[:4])

    rotated, rotated_box, points, _ = _align_face(image, face)
    (rx, ry), (lx, ly), _, _, _ = points
    eye_distance = math.hypot(lx - rx, ly - ry)

    symmetry = _symmetry_score(points)
    harmony = _harmony_score(points, *rotated_box, eye_distance)
    jaw = _jaw_score(rotated, rotated_box, points)

    # A stable crop around the nose is used for skin, hair and lighting.
    crop_center_x = float(points[2][0])
    crop_center_y = rotated_box[1] + rotated_box[3] * 0.50
    half_width = rotated_box[2] * 0.68
    half_height = rotated_box[3] * 0.62
    x0 = int(max(0, round(crop_center_x - half_width)))
    y0 = int(max(0, round(crop_center_y - half_height)))
    x1 = int(min(width, round(crop_center_x + half_width)))
    y1 = int(min(height, round(crop_center_y + half_height)))
    crop = rotated[y0:y1, x0:x1]
    if crop.size == 0:
        raise FaceNotFoundError(
            "Лицо слишком близко к краю кадра. Поверни камеру и попробуй снова."
        )
    crop = cv2.resize(crop, (200, 250), interpolation=cv2.INTER_AREA)

    skin = _skin_score(crop)
    hair = _hair_score(rotated, *rotated_box)
    light, brightness = _light_score(crop)

    score = _compose_score(symmetry, jaw, harmony, skin, hair, light, seed)
    tier = tier_for_score(score)

    stats = (
        Stat("jaw", "ЧЕЛЮСТЬ", round(jaw)),
        Stat("symmetry", "СИММЕТРИЯ", round(symmetry)),
        Stat("harmony", "ПРОПОРЦИИ", round(harmony)),
        Stat("skin", "КОЖА", round(skin)),
        Stat("hair", "ОБЪЁМ", round(hair)),
        Stat("light", "СВЕТ", round(light)),
    )

    face_area_ratio = (fw * fh) / max(1.0, width * height)
    notes = _notes_for(tier, stats, face_area_ratio, brightness)

    return Analysis(
        score=round(score, 1),
        tier=tier,
        stats=stats,
        notes=notes,
        face_box=(
            round(x / width, 4),
            round(y / height, 4),
            round(fw / width, 4),
            round(fh / height, 4),
        ),
        confidence=round(confidence, 4),
    )


def analyze_data_url(data_url: str) -> Analysis:
    return analyze_bytes(decode_data_url(data_url))
