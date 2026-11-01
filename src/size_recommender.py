"""
size_recommender.py — Rule-based size & fit recommendation.
No ML model needed: uses a standard height chart adjusted by build.
"""

SIZE_CHART_CM = [
    ("XS", 0, 155),
    ("S", 155, 162),
    ("M", 162, 168),
    ("L", 168, 174),
    ("XL", 174, 180),
    ("XXL", 180, 999),
]

_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]

BUILD_ADJUST = {
    "slim": -1,
    "medium": 0,
    "athletic": 0,
    "heavy / broad": 1,
}


def height_from_feet_inches(feet: int, inches: int) -> float:
    total_inches = feet * 12 + inches
    return round(total_inches * 2.54, 1)


def recommend_size(height_cm: float, build: str, category: str = "Top") -> dict:
    build_key = (build or "medium").lower()

    base_size = "M"
    for size, lo, hi in SIZE_CHART_CM:
        if lo <= height_cm < hi:
            base_size = size
            break

    idx = _ORDER.index(base_size)
    idx += BUILD_ADJUST.get(build_key, 0)
    idx = max(0, min(len(_ORDER) - 1, idx))
    final_size = _ORDER[idx]

    note = (
        f"Based on a height of **{height_cm:.0f} cm** and a **{build}** build, "
        f"we'd recommend size **{final_size}**"
    )
    note += f" for {category.lower()}s." if category else "."
    note += (
        " Sizing can vary a little between brands — if you're between two "
        "sizes, sizing up is usually the safer bet for a comfortable fit."
    )
    return {"size": final_size, "note": note}
