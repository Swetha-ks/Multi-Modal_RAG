"""
sizes.py — Assigns realistic size options (with price adjustments) to
each product, based on its category. The ABO/sample catalog doesn't
carry size data, so sizes are derived deterministically from the
product_id (same product always gets the same sizes — stable across
reruns, no randomness bugs).
"""

import hashlib

APPAREL_SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
SHOE_SIZES = ["6", "7", "8", "9", "10", "11"]
ONE_SIZE_CATEGORIES = {
    "bag", "accessory", "belt", "scarf", "saree", "sari",
}

# Extra amount (in ₹) added on top of the base price for sizes larger
# than M — mirrors how many real stores price plus-sizes slightly higher.
SIZE_SURCHARGE = {
    "XS": 0, "S": 0, "M": 0, "L": 50, "XL": 100, "XXL": 150,
    "6": 0, "7": 0, "8": 0, "9": 50, "10": 50, "11": 100,
}


def _stable_seed(product_id: str) -> int:
    return int(hashlib.md5(str(product_id).encode()).hexdigest(), 16)


def get_size_options(product: dict):
    """
    Returns a list of (size_label, price_for_this_size) tuples for the
    given product, based on its category.
    """
    category = str(product.get("category", "")).lower()
    base_price = product.get("price", 0)

    if any(key in category for key in ONE_SIZE_CATEGORIES):
        return [("One Size", base_price)]

    if "shoe" in category or "sneaker" in category or "boot" in category:
        seed = _stable_seed(product.get("product_id", ""))
        # Every shoe gets a contiguous 4-size run out of the 6 available
        start = seed % 3
        sizes = SHOE_SIZES[start:start + 4]
        return [(s, base_price + SIZE_SURCHARGE.get(s, 0)) for s in sizes]

    # Default: apparel sizing
    seed = _stable_seed(product.get("product_id", ""))
    start = seed % 2  # either starts at XS or S
    sizes = APPAREL_SIZES[start:start + 5]
    return [(s, base_price + SIZE_SURCHARGE.get(s, 0)) for s in sizes]


def default_size(product: dict):
    """Middle-ish size as the pre-selected default (feels more natural)."""
    options = get_size_options(product)
    return options[len(options) // 2]
