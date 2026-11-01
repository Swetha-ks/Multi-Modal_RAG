"""
seed_reviews.py — Populate the reviews table with realistic synthetic
reviews for every product in the catalog. Run once after indexing:

    python scripts/seed_reviews.py

Safe to re-run — it skips products that already have reviews.
"""
import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from src.user_store import UserStore

load_dotenv()

NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Kabir", "Meera",
    "Arjun", "Divya", "Rahul", "Isha", "Karan", "Neha", "Aditya", "Pooja",
    "Sanjay", "Ritika", "Manish", "Tanvi",
]

POSITIVE = [
    "Great quality for the price, fits true to size.",
    "Loved the material — feels premium and durable.",
    "Exactly as pictured, arrived quickly and well packed.",
    "Comfortable for all-day wear, would buy again.",
    "Color is even nicer in person. Highly recommend.",
]
NEUTRAL = [
    "Decent product, but sizing runs a little small.",
    "Good value overall, stitching could be better.",
    "Looks nice but took longer to arrive than expected.",
]
NEGATIVE = [
    "Material felt thinner than expected for the price.",
    "Color was slightly different from the photos.",
]


def synthetic_reviews_for(product) -> list:
    n = random.randint(3, 6)
    reviews = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.65:
            rating, pool = random.choice([4, 5]), POSITIVE
        elif roll < 0.9:
            rating, pool = 3, NEUTRAL
        else:
            rating, pool = random.choice([1, 2]), NEGATIVE
        reviews.append({
            "reviewer_name": random.choice(NAMES),
            "rating": rating,
            "comment": random.choice(pool),
        })
    return reviews


def main():
    random.seed(42)
    catalog_path = os.getenv("CATALOG_PATH", "./data/abo_products.csv")
    df = pd.read_csv(catalog_path)

    store = UserStore(os.getenv("USER_DB_PATH", "./data/users.db"))

    seeded, skipped = 0, 0
    for _, row in df.iterrows():
        pid = str(row["product_id"])
        if store.get_reviews(pid):
            skipped += 1
            continue
        for r in synthetic_reviews_for(row):
            store.add_review(pid, r["reviewer_name"], r["rating"], r["comment"])
        seeded += 1

    print(f"✅ Seeded reviews for {seeded} products ({skipped} already had reviews).")


if __name__ == "__main__":
    main()
