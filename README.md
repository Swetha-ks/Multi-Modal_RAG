# 🛍️ ShopAI — Multimodal RAG E-Commerce Assistant (Capstone Version)

A shopping assistant that combines **CLIP multimodal embeddings**, **ChromaDB**
vector search, and **LLaMA 3 (via Groq)** in a Retrieval-Augmented Generation
pipeline — wrapped in a full e-commerce experience with accounts, cart,
orders, and analytics.

## Dataset

Built on the **real Amazon Berkeley Objects (ABO) dataset**:
- Official page: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- Registry: https://registry.opendata.aws/amazon-berkeley-objects/
- Citation: Collins et al., *"ABO: Dataset and Benchmarks for Real-World 3D
  Object Understanding"*, CVPR 2022. https://arxiv.org/abs/2110.06199

`scripts/download_abo_full.py` downloads the official ABO metadata
archive (~1.3 GB, resumable if interrupted) and then parses out however
many real, valid, English-named products you ask for via `--scale`:

| Scale | Products | Approx. index time (text-only) |
|---|---|---|
| `tiny` (default) | 300 | ~1-2 min |
| `small` | 5,000 | ~10-20 min |
| `medium` | 50,000 | ~1-2 hours |
| `full` | 147,702 (all) | ~4-6 hours |

The archive download itself is the same size regardless of `--scale`
(it's one bundle containing all products' metadata) — `--scale` only
controls how many records get parsed and kept afterward. If your
connection drops mid-download, just re-run the same command; it
resumes from the last byte instead of starting over.

## Features

| Category | Features |
|---|---|
| **Search** | Text, image upload, and voice (🎤 speak / 🔊 spoken replies), category browse + keyword search |
| **Accounts** | Real login with PBKDF2-hashed passwords, per-user cart/wishlist/orders/history (SQLite) |
| **Shopping** | Cart, Wishlist (move-to-cart), demo Checkout with **coupon codes**, simulated **Order Tracking** (status auto-progresses over time) |
| **Discovery** | Product **Comparison** (up to 4 side-by-side), "**You might also like**" related items, **Recently Viewed** |
| **Personal Fit** | **Size & Fit Recommender** (height + build → size), **Outfit Builder** (upload a photo, find matching pieces) |
| **Trust signals** | **Reviews & star ratings** (seeded + user-submitted) shown on every product |
| **Utility** | **Price Drop Alerts** (checked each visit + manual recheck, plus a demo "simulate drop" button), **Chat History → PDF export** |
| **Checkout** | 3-step real checkout: **map-based address picker** (free, OpenStreetMap — click to set delivery location, auto-reverse-geocoded to a real address) → **OTP verification** (real SMS via your own Twilio account, or on-screen demo mode if not configured) → **Payment method** (Cash on Delivery, UPI with a demo transaction reference, or Cash) |
| **Order Tracking** | 5-stage visual stepper (Placed → Packed → Shipped → Out for Delivery → Delivered) **plus a real map** showing a route line from a fixed fulfillment center to your delivery address, with a marker positioned along it by delivery progress |
| **User Dashboard** | **📈 Trends** page: real "Trending Now" (aggregated from actual view activity across all shoppers), a price trend + forecast chart per category (seeded simulation — see honesty note below), and personalized "Recommended for You" based on your recently viewed items |
| **Admin Dashboard** | **🛠️ Admin** page (log in with username `admin`): platform-wide revenue/users/orders metrics, order-status breakdown, signups-over-time chart, best-selling products, a full orders table across all users, and active coupon codes |
| **Evaluation** | **Analytics dashboard**: catalog charts (category/price/brand) + a retrieval-quality evaluation (Precision@k, category hit rate, mean similarity) against a fixed 18-query test set — useful for the project report |

### Honesty note on the Trends dashboard
There's no real historical price record for this catalog (the dataset is a single price snapshot), so the price trend/forecast chart is a **deterministic random-walk simulation** anchored to today's real average price — clearly labeled as such in the UI. The "Trending Now" and "Best-Selling Products" sections, by contrast, are **real aggregates** from actual logged views and placed orders in your database, not simulated.

### Getting admin access
Register a new account with the username `admin` (any password) — this reserved username is automatically granted admin rights, giving you the **🛠️ Admin** nav button. Every other username is a normal shopper account.

## Setup

> `requirements.txt` now also installs `folium`, `streamlit-folium`, and `geopy` (free map + address lookup — no API key) and `twilio` (optional, only used if you configure real SMS). No extra setup needed unless you want real SMS — see `.env.example`.

```bash
# 1. Install PyTorch
pip install torch torchvision

# 2. Install everything else
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env      # then paste your free Groq API key (console.groq.com)

# 4. Get the real dataset — tiny scale is fast and plenty for a demo
python scripts/download_abo_full.py --scale tiny
# (or use the bundled 30-item curated sample instead: python scripts/download_abo.py)

# 5. Index the catalog (fast by default — text-only embeddings)
python scripts/index_catalog.py

# 6. Seed demo reviews
python scripts/seed_reviews.py

# 7. Run
streamlit run app.py
```
Open http://localhost:8501, create an account, and start shopping.

### If you're short on time
`--scale tiny` (300 real products) is genuinely enough for a strong demo —
real names, real categories, real images, multiple categories, and every
one of the 10 features works identically regardless of catalog size.
There's no need to reach for `--scale full` unless your project
specifically requires the complete dataset; it costs hours for a download
and re-index that doesn't change what any feature demonstrates.

### Why indexing is fast by default
`index_catalog.py` embeds **product text only** (name, description,
color, category, brand, price) with CLIP by default — it does not
download and CLIP-encode every product image during indexing. That
network+compute step is what makes naive multimodal indexing pipelines
slow. Product photos still render normally in the UI (Streamlit loads
`image_url` directly), and a shopper's own uploaded photo is still fully
CLIP-embedded live at query time — multimodal search is preserved where
it matters most. Pass `--with-images` to `index_catalog.py` if you want
true image+text indexing (slower, best for small catalogs).

## Project Structure
```
rag_final/
├── app.py                      ← Full UI: login, chat, browse, cart, wishlist,
│                                   orders, alerts, compare, size, outfit, analytics
├── requirements.txt
├── .env.example
├── data/
│   └── abo_products.csv        ← real ABO fashion products (or your sample)
│   └── users.db                 ← created on first run (accounts, cart, orders, reviews...)
├── src/
│   ├── embedder.py              ← CLIP
│   ├── database.py              ← ChromaDB vector search (fast text-only indexing)
│   ├── pipeline.py               ← RAG (retrieve + LLaMA 3 generate)
│   ├── voice.py                  ← Voice input/output
│   ├── auth.py                   ← Password hashing (PBKDF2, stdlib only)
│   ├── user_store.py             ← SQLite: users, cart, wishlist, alerts, history, orders, reviews, recently-viewed
│   ├── orders.py                  ← Coupon codes + simulated order status
│   ├── recommender.py             ← "You might also like" related products
│   ├── size_recommender.py        ← Rule-based size & fit logic
│   └── pdf_export.py              ← Chat → PDF export
├── scripts/
│   ├── download_abo.py            ← 30-item curated sample (instant, offline)
│   ├── download_abo_full.py       ← Real ABO dataset via --scale tiny/small/medium/full
│   ├── index_catalog.py           ← Index products into ChromaDB (--with-images to opt into multimodal)
│   └── seed_reviews.py             ← Seed synthetic reviews for demo
└── evaluation/
    └── evaluate.py                 ← RAG retrieval evaluation (Precision@k etc.)
```

## Notes for the Report
- Passwords: PBKDF2-HMAC-SHA256, 260,000 iterations, random per-user salt — no plaintext storage, no extra crypto dependency.
- Per-user data (cart/wishlist/alerts/orders/reviews/history) lives in `data/users.db`, separate from the product vector store (`chroma_db/`).
- The Analytics tab's evaluation metrics (Precision@k, category hit rate, mean CLIP similarity) are computed live against a fixed test-query set in `evaluation/evaluate.py` — cite this as your retrieval-quality evaluation methodology.
- Order status ("Placed → Packed → Shipped → Delivered") auto-progresses based on elapsed time since checkout, purely for demo purposes — no real logistics integration.
