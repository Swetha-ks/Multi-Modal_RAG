"""
download_abo_full.py
────────────────────
Downloads and parses the REAL Amazon Berkeley Objects (ABO) dataset
directly from Amazon's official, public S3 bucket (no API key needed —
it's an open dataset).

Dataset info:
  Official page:  https://amazon-berkeley-objects.s3.amazonaws.com/index.html
  Registry:       https://registry.opendata.aws/amazon-berkeley-objects/
  Paper:          Collins et al., "ABO: Dataset and Benchmarks for
                   Real-World 3D Object Understanding", CVPR 2022.
                   https://arxiv.org/abs/2110.06199

147,702 real Amazon products with images, names, categories, colors,
materials, and (where available) prices.

USAGE (fast → slow)
────────────────────
    python scripts/download_abo_full.py --scale tiny       # 300 products   (~1-2 min to index)
    python scripts/download_abo_full.py --scale small      # 5,000 products (~10-20 min to index)
    python scripts/download_abo_full.py --scale medium     # 50,000 products (~1-2 hrs)
    python scripts/download_abo_full.py --scale full       # all 147,702 (~4-6 hrs)

⏱️ TIGHT ON TIME? Use --scale tiny. It's still 100% real ABO data
(real product names, real categories, real images from Amazon's
official archive) — just fewer of them, so indexing finishes fast.

Runs non-interactively (safe for scripts/CI) and RESUMES an interrupted
download automatically (uses HTTP Range requests against the partially
downloaded .tar file), so a dropped connection doesn't cost you the
whole download again. Note: the listings archive itself (~1.3 GB) is
the same size regardless of --scale, since --scale only controls how
many records get PARSED out of it afterward — the archive holds ALL
147,702 records' text metadata bundled together.
"""

import os
import sys
import json
import gzip
import tarfile
import argparse
import requests
import pandas as pd
from tqdm import tqdm

ABO_LISTINGS_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"
)
ABO_IMAGE_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small"

OUTPUT_PATH = "./data/abo_products.csv"
DOWNLOAD_PATH = "./data/abo-listings.tar"

SCALE_LIMITS = {"tiny": 300, "small": 5_000, "medium": 50_000, "full": None}


def get_english_value(field_list) -> str:
    if not field_list:
        return ""
    if isinstance(field_list, str):
        return field_list
    for item in field_list:
        if isinstance(item, dict):
            lang = item.get("language_tag", "")
            if lang.startswith("en"):
                return str(item.get("value", ""))
    if isinstance(field_list[0], dict):
        return str(field_list[0].get("value", ""))
    return str(field_list[0])


def get_price(price_field) -> int:
    if not price_field:
        return 0
    if isinstance(price_field, list) and len(price_field) > 0:
        p = price_field[0]
        if isinstance(p, dict):
            val = p.get("value", 0)
            try:
                usd = float(str(val).replace(",", ""))
                return int(usd * 83)  # rough USD → INR
            except Exception:
                return 999
    return 999


def get_image_url(image_id: str) -> str:
    if not image_id:
        return ""
    prefix = image_id[:2]
    return f"{ABO_IMAGE_BASE}/{prefix}/{image_id}.jpg"


def download_listings(force: bool = False) -> bool:
    """
    Downloads the ABO listings tar, RESUMING a partial download if one
    exists (via HTTP Range header) instead of starting over.
    """
    os.makedirs("./data", exist_ok=True)

    if os.path.exists(DOWNLOAD_PATH) and not force:
        try:
            head = requests.head(ABO_LISTINGS_URL, timeout=30)
            remote_size = int(head.headers.get("content-length", 0))
            local_size = os.path.getsize(DOWNLOAD_PATH)
            if remote_size and local_size >= remote_size:
                print(f"✅ Already fully downloaded at {DOWNLOAD_PATH}")
                return True
            print(f"↻ Resuming download from byte {local_size:,} of {remote_size:,}")
        except Exception:
            print("⚠️ Could not verify remote size; will attempt to resume anyway.")
            local_size = os.path.getsize(DOWNLOAD_PATH)
    else:
        local_size = 0

    headers = {"Range": f"bytes={local_size}-"} if local_size else {}
    mode = "ab" if local_size else "wb"

    print("📥 Downloading ABO listings from Amazon S3 (official public dataset)...")
    print(f"   Source: {ABO_LISTINGS_URL}")
    print("   Full archive is ~1.3 GB (contains ALL 147,702 records' metadata,")
    print("   regardless of --scale) — this can take a while on slower connections.\n")

    try:
        response = requests.get(ABO_LISTINGS_URL, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) + local_size

        with open(DOWNLOAD_PATH, mode) as f:
            with tqdm(total=total, initial=local_size, unit="B", unit_scale=True,
                      desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"\n✅ Download complete: {DOWNLOAD_PATH}")
        return True

    except requests.exceptions.ConnectionError:
        print("\n❌ Connection failed. Your progress is saved — re-run this script "
              "to resume the download once your connection is back.")
        return False
    except Exception as e:
        print(f"\n❌ Download failed: {e}. Progress saved — re-run to resume.")
        return False


def parse_listings(max_products=None) -> pd.DataFrame:
    print("\n📂 Parsing ABO listings...")
    products, skipped = [], 0

    with tarfile.open(DOWNLOAD_PATH, "r") as tar:
        json_files = [m for m in tar.getmembers() if m.name.endswith(".json.gz")]
        print(f"   Found {len(json_files):,} listing files to process")

        for json_file in tqdm(json_files, desc="Processing files"):
            if max_products and len(products) >= max_products:
                break
            try:
                f = tar.extractfile(json_file)
                if not f:
                    continue
                with gzip.open(f, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        if max_products and len(products) >= max_products:
                            break
                        try:
                            item = json.loads(line.strip())
                            product_id = item.get("item_id", "")
                            if not product_id:
                                skipped += 1
                                continue

                            name = get_english_value(item.get("item_name", []))
                            if not name or len(name) < 3:
                                skipped += 1
                                continue

                            category = get_english_value(item.get("product_type", [])) or "General"
                            category = category.replace("_", " ").title()

                            bullets = item.get("bullet_point", [])
                            desc_parts = []
                            for b in bullets[:3]:
                                val = get_english_value(b if isinstance(b, list) else [b])
                                if val:
                                    desc_parts.append(val)
                            description = ". ".join(desc_parts) or name

                            color = get_english_value(item.get("color", [])) or "Not specified"
                            material = get_english_value(item.get("material", []))
                            brand = get_english_value(item.get("brand", []))

                            price = get_price(item.get("price", [])) or 999

                            image_url = get_image_url(item.get("main_image_id", ""))

                            products.append({
                                "product_id": product_id,
                                "name": name[:200],
                                "category": category,
                                "description": description[:500],
                                "price": price,
                                "color": color,
                                "material": material,
                                "brand": brand,
                                "stock": "In Stock",
                                "image_url": image_url,
                            })
                        except (json.JSONDecodeError, Exception):
                            skipped += 1
            except Exception as e:
                print(f"   Error reading {json_file.name}: {e}")
                continue

    print(f"\n✅ Parsed {len(products):,} products ({skipped:,} skipped)")
    return pd.DataFrame(products)


def save_dataset(df: pd.DataFrame):
    os.makedirs("./data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to {OUTPUT_PATH}")
    print("\n📊 Dataset Summary:")
    print(f"   Total products: {len(df):,}")
    print(f"   Categories:     {df['category'].nunique()}")
    print(f"   Brands:         {df['brand'].nunique()}")
    print(f"\n   Top 10 categories:")
    for cat, count in df["category"].value_counts().head(10).items():
        print(f"      {cat}: {count:,}")
    print(f"\n   Price range: ₹{df['price'].min():,} - ₹{df['price'].max():,} "
          f"(avg ₹{df['price'].mean():,.0f})")


def main():
    parser = argparse.ArgumentParser(description="Download & prepare the real ABO dataset.")
    parser.add_argument("--scale", choices=["tiny", "small", "medium", "full"], default="tiny",
                         help="tiny=300 (fastest) | small=5,000 | medium=50,000 | full=147,702")
    parser.add_argument("--force-redownload", action="store_true",
                         help="Ignore any partially-downloaded file and start fresh.")
    args = parser.parse_args()

    max_p = SCALE_LIMITS[args.scale]

    print("=" * 60)
    print("  ABO Real Dataset Downloader")
    print("  Amazon Berkeley Objects — 147,702 real products")
    print(f"  Scale: {args.scale} ({max_p or 'ALL'} products)")
    print("=" * 60)

    if not download_listings(force=args.force_redownload):
        print("\n⚠️  Download incomplete. Re-run this exact command to resume:")
        print(f"   python {os.path.basename(__file__)} --scale {args.scale}")
        sys.exit(1)

    df = parse_listings(max_products=max_p)
    if df.empty:
        print("❌ No products parsed. Please check the downloaded archive.")
        sys.exit(1)

    save_dataset(df)
    print("\n" + "=" * 60)
    print(f"  ✅ Done! {len(df):,} real ABO products ready.")
    print("  Next: python scripts/index_catalog.py --text-only   (fast)")
    print("=" * 60)


if __name__ == "__main__":
    main()
