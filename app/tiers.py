"""The seven-tier ladder from the original meme sheet."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tier:
    id: str
    name: str
    tag: str
    description: str
    color: str
    emoji: str
    max_score: float
    rarity: str


TIERS: tuple[Tier, ...] = (
    Tier(
        id="sub3",
        name="SUB 3",
        tag="гены сказали «позже»",
        description=(
            "Камера сегодня не на твоей стороне. Но у тебя точно есть харизма — "
            "просто она осталась за кадром."
        ),
        color="#FE2C55",
        emoji="💀",
        max_score=24.9,
        rarity="1 из 3",
    ),
    Tier(
        id="sub5",
        name="SUB 5",
        tag="почти нормис",
        description=(
            "База есть. Осталось найти свет, ракурс и барбера, который не подведёт."
        ),
        color="#FF7A00",
        emoji="😬",
        max_score=39.9,
        rarity="1 из 6",
    ),
    Tier(
        id="ltn",
        name="LTN",
        tag="low tier normie",
        description=(
            "Обычный человек. Не звезда алгоритма, но и не повод для паники. "
            "Апгрейд вполне реален."
        ),
        color="#FFC400",
        emoji="😐",
        max_score=54.9,
        rarity="1 из 3",
    ),
    Tier(
        id="mtn",
        name="MTN",
        tag="mid tier normie",
        description=(
            "Крепкая середина. Ты уже норм — осталось перестать снимать снизу "
            "и поверить в боковой свет."
        ),
        color="#B8FF3C",
        emoji="🙂",
        max_score=69.9,
        rarity="1 из 2",
    ),
    Tier(
        id="htn",
        name="HTN",
        tag="high tier normie",
        description=(
            "Хорош. С таким лицом можно не прятаться за фильтрами. "
            "Но можно и продолжить — алгоритм не против."
        ),
        color="#25F4EE",
        emoji="😎",
        max_score=81.9,
        rarity="1 из 8",
    ),
    Tier(
        id="chad",
        name="CHAD",
        tag="челюсть режет стекло",
        description=(
            "Челюсть режет стекло, волосы живут своей жизнью. "
            "Пожалуйста, не продавай курсы по успеху."
        ),
        color="#7C5CFF",
        emoji="🗿",
        max_score=92.9,
        rarity="1 из 25",
    ),
    Tier(
        id="true_adam",
        name="TRUE ADAM",
        tag="легенда",
        description=(
            "Идеал, которого не существует. Камера считает, что ты — исключение. "
            "Скриншот в рамку, мем в историю."
        ),
        color="#FFD76A",
        emoji="👑",
        max_score=100.0,
        rarity="1 из 1000",
    ),
)


def tier_for_score(score: float) -> Tier:
    """Return the tier whose inclusive upper bound contains *score*."""
    normalized = max(0.0, min(100.0, float(score)))
    for tier in TIERS:
        if normalized <= tier.max_score:
            return tier
    return TIERS[-1]
