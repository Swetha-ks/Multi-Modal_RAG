"""
index_catalog.py — indexes the product catalog (sample or real ABO) into ChromaDB.

USAGE
─────
    python scripts/index_catalog.py                  # FAST (default): text-only embeddings
    python scripts/index_catalog.py --with-images     # slower: also embeds each product image

Text-only is the default because the per-product image download is what
makes indexing slow (one HTTP request per product). Product photos still
render normally in the UI either way — this only affects whether the
image itself is fed into the CLIP embedding used for search ranking.

RESUMABLE: already-indexed product_ids are skipped automatically, so
if the process is interrupted you can just re-run the same command
and it will pick up where it left off — no progress is lost.
"""
import sys, os, argparse, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.embedder import MultimodalEmbedder
from src.database import ProductDatabase
from dotenv import load_dotenv
load_dotenv()

BATCH_SIZE = 100


def main():
    parser = argparse.ArgumentParser(description="Index the product catalog into ChromaDB.")
    parser.add_argument("--with-images", action="store_true",
                         help="Also download+embed each product image (slower, true multimodal).")
    parser.add_argument("--reindex", action="store_true",
                         help="Wipe the existing index and start fresh (non-interactive).")
    args = parser.parse_args()
    text_only = not args.with_images

    print("=" * 60)
    print("  Catalog Indexing" + ("  (text-only, fast mode)" if text_only else "  (multimodal: text+images)"))
    print("=" * 60)

    path = os.getenv("CATALOG_PATH", "./data/abo_products.csv")
    if not os.path.exists(path):
        print(f"Dataset not found at {path}")
        print("Run: python scripts/download_abo.py   (sample)")
        print("or:  python scripts/download_abo_full.py --scale tiny   (real ABO, fast)")
        return

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} products from {path}")

    embedder = MultimodalEmbedder()
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    db = ProductDatabase(db_path=db_path)

    if args.reindex and db.get_count() > 0:
        db.client.delete_collection("abo_products")
        db.collection = db.client.create_collection(
            name="abo_products", metadata={"hnsw:space": "cosine"}
        )
        print("♻️  Existing index cleared (--reindex).")

    # RESUME SUPPORT: skip product_ids already indexed
    already_indexed = set()
    if db.get_count() > 0:
        existing = db.collection.get(include=[])
        already_indexed = set(existing.get("ids", []))
        print(f"↻ Found {len(already_indexed):,} already indexed — will skip those.")

    df = df[~df["product_id"].astype(str).isin(already_indexed)]
    if df.empty:
        print("✅ Nothing new to index — catalog is already fully indexed.")
        return

    print(f"\nIndexing {len(df):,} new products in batches of {BATCH_SIZE}"
          f"{' (text-only)' if text_only else ' (with images)'}...\n")

    batches = [df[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    start = time.time()

    for batch in tqdm(batches, desc="Indexing"):
        ids, embeddings, metadatas, documents = [], [], [], []
        for _, row in batch.iterrows():
            try:
                text = (f"{row['name']}. {row['description']} "
                        f"Color: {row['color']}. Category: {row['category']}. "
                        f"Brand: {row.get('brand', '')}. Price: Rs.{row['price']}.")

                img_emb = None
                if not text_only and pd.notna(row.get("image_url")) and \
                        str(row["image_url"]).startswith("http"):
                    img_emb = embedder.embed_image_from_url(str(row["image_url"]))

                if img_emb:
                    t = np.array(embedder.embed_text(text))
                    i = np.array(img_emb)
                    c = (t + i) / 2
                    final = (c / np.linalg.norm(c)).tolist()
                else:
                    final = embedder.embed_text(text)

                ids.append(str(row["product_id"]))
                embeddings.append(final)
                documents.append(text[:1000])
                metadatas.append({
                    "product_id": str(row["product_id"]),
                    "name": str(row["name"])[:200],
                    "category": str(row["category"]),
                    "description": str(row["description"])[:500],
                    "price": int(row["price"]),
                    "color": str(row["color"]),
                    "material": str(row.get("material", "")),
                    "brand": str(row.get("brand", "")),
                    "stock": str(row.get("stock", "In Stock")),
                    "image_url": str(row.get("image_url", "")),
                })
            except Exception:
                continue

        if ids:
            try:
                db.collection.add(ids=ids, embeddings=embeddings,
                                   documents=documents, metadatas=metadatas)
            except Exception as e:
                print(f"Batch error (progress so far is safe): {e}")

    elapsed = time.time() - start
    print(f"\n✅ Indexed {db.get_count():,} products total in {elapsed/60:.1f} min!")
    print("Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
