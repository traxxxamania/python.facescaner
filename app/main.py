"""FastAPI application: JSON API + the static TIER CHECK frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__, analyzer, scoring
from .tiers import TIERS

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_BODY_BYTES = 9_000_000


@asynccontextmanager
async def lifespan(_: FastAPI):
    analyzer.warmup()
    scoring.warmup()
    yield


app = FastAPI(
    title="TIER CHECK",
    version=__version__,
    description=(
        "Мем-сканер лица: FastAPI + OpenCV YuNet + обученная NumPy-модель оценки. "
        "Оценка не является реальной оценкой привлекательности."
    ),
    lifespan=lifespan,
)


class RateRequest(BaseModel):
    image: str = Field(
        min_length=64,
        max_length=12_000_000,
        description="Data URL или base64-строка с JPG/PNG/WEBP.",
    )


class TierResponse(BaseModel):
    id: str
    name: str
    tag: str
    description: str
    color: str
    emoji: str
    max_score: float
    rarity: str


class StatResponse(BaseModel):
    key: str
    label: str
    value: int


class FaceBoxResponse(BaseModel):
    x: float
    y: float
    width: float
    height: float


class RateResponse(BaseModel):
    score: float
    tier: TierResponse
    stats: list[StatResponse]
    notes: list[str]
    face_box: FaceBoxResponse
    confidence: float
    scoring_model: str
    disclaimer: str


@app.middleware("http")
async def add_headers_and_limit_body(request: Request, call_next):
    if request.method == "POST":
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                too_large = int(raw_length) > MAX_BODY_BYTES
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Слишком большой запрос. Максимум 8 МБ."},
                )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(analyzer.InvalidImageError)
async def invalid_image_handler(_: Request, exc: analyzer.InvalidImageError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(analyzer.FaceNotFoundError)
async def face_not_found_handler(_: Request, exc: analyzer.FaceNotFoundError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Сканер не Ready. Проверь, что модели лежат в app/models/."
        },
    )


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": __version__,
        "model": "face_detection_yunet_2026may.onnx",
        "scoring_model": scoring.MODEL_VERSION,
        "scoring_rubric": scoring.RUBRIC_VERSION,
    }


@app.get("/api/tiers", response_model=list[TierResponse], tags=["tiers"])
def list_tiers() -> list[TierResponse]:
    return [
        TierResponse(
            id=tier.id,
            name=tier.name,
            tag=tier.tag,
            description=tier.description,
            color=tier.color,
            emoji=tier.emoji,
            max_score=tier.max_score,
            rarity=tier.rarity,
        )
        for tier in TIERS
    ]


@app.post("/api/rate", response_model=RateResponse, tags=["scan"])
def rate_photo(payload: RateRequest):
    result = analyzer.analyze_data_url(payload.image)
    return RateResponse(
        score=result.score,
        tier=TierResponse(
            id=result.tier.id,
            name=result.tier.name,
            tag=result.tier.tag,
            description=result.tier.description,
            color=result.tier.color,
            emoji=result.tier.emoji,
            max_score=result.tier.max_score,
            rarity=result.tier.rarity,
        ),
        stats=[
            StatResponse(key=stat.key, label=stat.label, value=stat.value)
            for stat in result.stats
        ],
        notes=list(result.notes),
        face_box=FaceBoxResponse(
            x=result.face_box[0],
            y=result.face_box[1],
            width=result.face_box[2],
            height=result.face_box[3],
        ),
        confidence=result.confidence,
        scoring_model=scoring.MODEL_VERSION,
        disclaimer=(
            "Это мем-приложение. Оценка основана на геометрии кадра, "
            "освещении и текстуре — а не на реальной привлекательности."
        ),
    )


app.frontend("/", directory=STATIC_DIR, check_dir=True)
