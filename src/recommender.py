"""
recommender.py — "You might also like" content-based recommendations.
Uses the same category + a nearby price band, so it works instantly
off the already-indexed catalog metadata without extra model calls.
"""


def get_related_products(database, product: dict, k: int = 4):
    try:
        all_meta = database.collection.get(include=["metadatas"])["metadatas"]
    except Exception:
        return []

    same_category = [
        m for m in all_meta
        if m.get("category") == product.get("category")
        and str(m.get("product_id")) != str(product.get("product_id"))
    ]
    if not same_category:
        return []

    target_price = product.get("price", 0)
    same_category.sort(key=lambda m: abs(m.get("price", 0) - target_price))
    return same_category[:k]
