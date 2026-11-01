"""
user_store.py — SQLite-backed store for per-user data
────────────────────────────────────────────────────────
Handles: user accounts (hashed passwords), cart, wishlist,
price-drop alerts, chat history, orders, reviews, and recently
viewed products. Each user only ever sees their own rows (scoped
by user_id), giving every account its own persistent state across
sessions.
"""

import sqlite3
import json
import threading
from datetime import datetime
from contextlib import contextmanager

from src.auth import hash_password, verify_password


class UserStore:

    def __init__(self, db_path: str = "./data/users.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS cart (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                product_json TEXT NOT NULL,
                added_at TEXT NOT NULL,
                UNIQUE(user_id, product_id),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS wishlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                product_json TEXT NOT NULL,
                added_at TEXT NOT NULL,
                UNIQUE(user_id, product_id),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                product_json TEXT NOT NULL,
                threshold_price REAL NOT NULL,
                created_at TEXT NOT NULL,
                triggered INTEGER DEFAULT 0,
                triggered_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                products_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                items_json TEXT NOT NULL,
                subtotal REAL NOT NULL,
                discount REAL NOT NULL DEFAULT 0,
                coupon_code TEXT,
                total REAL NOT NULL,
                placed_at TEXT NOT NULL,
                recipient_name TEXT,
                shipping_address TEXT,
                phone_number TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                user_id INTEGER,
                reviewer_name TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS recently_viewed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                product_json TEXT NOT NULL,
                viewed_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )""")
            self._migrate_add_columns(c, "orders",
                                       ["recipient_name", "shipping_address", "phone_number",
                                        "delivery_lat", "delivery_lng", "payment_method",
                                        "payment_reference"])
            self._migrate_add_columns(c, "users", ["is_admin"])

    def _migrate_add_columns(self, conn, table: str, columns: list):
        """Adds columns to a table that already existed before they were
        introduced, so older users.db files upgrade automatically instead
        of breaking on the new schema."""
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col in columns:
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")

    # ───────────────────────── Auth ─────────────────────────
    def create_user(self, username: str, password: str):
        username = username.strip()
        pwd_hash, salt = hash_password(password)
        # The reserved username "admin" automatically gets admin access —
        # a simple, demoable way to reach the admin dashboard without a
        # separate invite/promotion system.
        is_admin = 1 if username.lower() == "admin" else 0
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO users (username, password_hash, salt, created_at, is_admin) "
                    "VALUES (?,?,?,?,?)",
                    (username, pwd_hash, salt, datetime.utcnow().isoformat(), is_admin),
                )
            return True, "Account created — you can log in now."
        except sqlite3.IntegrityError:
            return False, "That username is already taken."

    def authenticate(self, username: str, password: str):
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM users WHERE username=?", (username.strip(),)
            ).fetchone()
        if not row:
            return None
        if verify_password(password, row["password_hash"], row["salt"]):
            try:
                is_admin = bool(int(row["is_admin"] or 0))
            except (ValueError, TypeError):
                is_admin = False
            return {"id": row["id"], "username": row["username"], "created_at": row["created_at"],
                    "is_admin": is_admin}
        return None

    def change_password(self, user_id: int, old_password: str, new_password: str):
        with self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not row or not verify_password(old_password, row["password_hash"], row["salt"]):
            return False, "Current password is incorrect."
        new_hash, new_salt = hash_password(new_password)
        with self._conn() as c:
            c.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?",
                      (new_hash, new_salt, user_id))
        return True, "Password updated."

    def get_account_stats(self, user_id: int) -> dict:
        orders = self.get_orders(user_id)
        return {
            "orders_count": len(orders),
            "total_spent": round(sum(o["total"] for o in orders), 2),
            "cart_count": len(self.get_cart(user_id)),
            "wishlist_count": len(self.get_wishlist(user_id)),
            "active_alerts": len(self.get_price_alerts(user_id, active_only=True)),
            "reviews_written": self._count_reviews_by_user(user_id),
        }

    def _count_reviews_by_user(self, user_id: int) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) as n FROM reviews WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["n"] if row else 0

    # ───────────────────────── Cart ─────────────────────────
    def add_to_cart(self, user_id: int, product: dict):
        with self._conn() as c:
            c.execute(
                "INSERT INTO cart (user_id, product_id, product_json, added_at) "
                "VALUES (?,?,?,?) "
                "ON CONFLICT(user_id, product_id) DO UPDATE SET product_json=excluded.product_json",
                (user_id, str(product["product_id"]), json.dumps(product),
                 datetime.utcnow().isoformat()),
            )

    def remove_from_cart(self, user_id: int, product_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM cart WHERE user_id=? AND product_id=?",
                      (user_id, str(product_id)))

    def get_cart(self, user_id: int):
        with self._conn() as c:
            rows = c.execute(
                "SELECT product_json FROM cart WHERE user_id=? ORDER BY added_at DESC",
                (user_id,),
            ).fetchall()
        return [json.loads(r["product_json"]) for r in rows]

    def clear_cart(self, user_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM cart WHERE user_id=?", (user_id,))

    def cart_total(self, user_id: int) -> float:
        return sum(p.get("price", 0) for p in self.get_cart(user_id))

    # ─────────────────────── Wishlist ───────────────────────
    def add_to_wishlist(self, user_id: int, product: dict):
        with self._conn() as c:
            c.execute(
                "INSERT INTO wishlist (user_id, product_id, product_json, added_at) "
                "VALUES (?,?,?,?) "
                "ON CONFLICT(user_id, product_id) DO UPDATE SET product_json=excluded.product_json",
                (user_id, str(product["product_id"]), json.dumps(product),
                 datetime.utcnow().isoformat()),
            )

    def remove_from_wishlist(self, user_id: int, product_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM wishlist WHERE user_id=? AND product_id=?",
                      (user_id, str(product_id)))

    def get_wishlist(self, user_id: int):
        with self._conn() as c:
            rows = c.execute(
                "SELECT product_json FROM wishlist WHERE user_id=? ORDER BY added_at DESC",
                (user_id,),
            ).fetchall()
        return [json.loads(r["product_json"]) for r in rows]

    def move_to_cart(self, user_id: int, product_id: str):
        wishlist = self.get_wishlist(user_id)
        product = next((p for p in wishlist if str(p["product_id"]) == str(product_id)), None)
        if product:
            self.add_to_cart(user_id, product)
            self.remove_from_wishlist(user_id, product_id)
        return product

    # ────────────────────── Price Alerts ──────────────────────
    def add_price_alert(self, user_id: int, product: dict, threshold_price: float):
        with self._conn() as c:
            c.execute(
                "INSERT INTO price_alerts "
                "(user_id, product_id, product_json, threshold_price, created_at, triggered) "
                "VALUES (?,?,?,?,?,0)",
                (user_id, str(product["product_id"]), json.dumps(product),
                 float(threshold_price), datetime.utcnow().isoformat()),
            )

    def get_price_alerts(self, user_id: int, active_only: bool = False):
        with self._conn() as c:
            q = "SELECT * FROM price_alerts WHERE user_id=?"
            if active_only:
                q += " AND triggered=0"
            q += " ORDER BY created_at DESC"
            rows = c.execute(q, (user_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["product"] = json.loads(r["product_json"])
            out.append(d)
        return out

    def remove_price_alert(self, user_id: int, alert_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM price_alerts WHERE user_id=? AND id=?",
                      (user_id, alert_id))

    def simulate_price_drop(self, user_id: int, alert_id: int):
        """
        Demo-only helper: marks a specific alert as triggered immediately,
        for testing/demonstrating the feature without needing to actually
        re-index the catalog with a lower price.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM price_alerts WHERE user_id=? AND id=?", (user_id, alert_id)
            ).fetchone()
            if not row:
                return None
            simulated_price = max(row["threshold_price"] - 1, 0)
            c.execute(
                "UPDATE price_alerts SET triggered=1, triggered_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), alert_id),
            )
            product = json.loads(row["product_json"])
            return {"product": product, "current_price": simulated_price,
                    "threshold_price": row["threshold_price"]}

    def check_alerts(self, user_id: int, current_price_lookup: dict):
        newly_triggered = []
        alerts = self.get_price_alerts(user_id, active_only=True)
        with self._conn() as c:
            for a in alerts:
                cur_price = current_price_lookup.get(str(a["product_id"]))
                if cur_price is not None and cur_price <= a["threshold_price"]:
                    c.execute(
                        "UPDATE price_alerts SET triggered=1, triggered_at=? WHERE id=?",
                        (datetime.utcnow().isoformat(), a["id"]),
                    )
                    a["current_price"] = cur_price
                    newly_triggered.append(a)
        return newly_triggered

    # ─────────────────────── Chat History ───────────────────────
    def save_message(self, user_id: int, role: str, content: str, products=None):
        with self._conn() as c:
            c.execute(
                "INSERT INTO chat_history (user_id, role, content, products_json, created_at) "
                "VALUES (?,?,?,?,?)",
                (user_id, role, content, json.dumps(products or []),
                 datetime.utcnow().isoformat()),
            )

    def get_history(self, user_id: int, limit: int = 300):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM chat_history WHERE user_id=? ORDER BY id ASC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "role": r["role"],
                "content": r["content"],
                "products": json.loads(r["products_json"] or "[]"),
                "created_at": r["created_at"],
            })
        return out

    def clear_history(self, user_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))

    # ────────────────────────── Orders ──────────────────────────
    def place_order(self, user_id: int, items: list, subtotal: float,
                     discount: float = 0.0, coupon_code: str = None,
                     recipient_name: str = None, shipping_address: str = None,
                     phone_number: str = None, delivery_lat: float = None,
                     delivery_lng: float = None, payment_method: str = None,
                     payment_reference: str = None):
        total = max(subtotal - discount, 0)
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO orders (user_id, items_json, subtotal, discount, coupon_code, total, "
                "placed_at, recipient_name, shipping_address, phone_number, delivery_lat, "
                "delivery_lng, payment_method, payment_reference) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, json.dumps(items), subtotal, discount, coupon_code, total,
                 datetime.utcnow().isoformat(), recipient_name, shipping_address, phone_number,
                 delivery_lat, delivery_lng, payment_method, payment_reference),
            )
            return cur.lastrowid

    def get_orders(self, user_id: int):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM orders WHERE user_id=? ORDER BY placed_at DESC", (user_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["items"] = json.loads(r["items_json"])
            out.append(d)
        return out

    # ────────────────────────── Reviews ──────────────────────────
    def add_review(self, product_id: str, reviewer_name: str, rating: int,
                    comment: str = "", user_id: int = None):
        rating = max(1, min(5, int(rating)))
        with self._conn() as c:
            c.execute(
                "INSERT INTO reviews (product_id, user_id, reviewer_name, rating, comment, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (str(product_id), user_id, reviewer_name, rating, comment,
                 datetime.utcnow().isoformat()),
            )

    def get_reviews(self, product_id: str):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM reviews WHERE product_id=? ORDER BY created_at DESC",
                (str(product_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def review_summary(self, product_id: str):
        reviews = self.get_reviews(product_id)
        if not reviews:
            return {"count": 0, "average": 0.0}
        avg = sum(r["rating"] for r in reviews) / len(reviews)
        return {"count": len(reviews), "average": round(avg, 1)}

    # ─────────────────────── Recently Viewed ───────────────────────
    def log_view(self, user_id: int, product: dict, keep_last: int = 12):
        pid = str(product["product_id"])
        with self._conn() as c:
            c.execute("DELETE FROM recently_viewed WHERE user_id=? AND product_id=?", (user_id, pid))
            c.execute(
                "INSERT INTO recently_viewed (user_id, product_id, product_json, viewed_at) "
                "VALUES (?,?,?,?)",
                (user_id, pid, json.dumps(product), datetime.utcnow().isoformat()),
            )
            ids_to_keep = [row["id"] for row in c.execute(
                "SELECT id FROM recently_viewed WHERE user_id=? ORDER BY viewed_at DESC LIMIT ?",
                (user_id, keep_last),
            ).fetchall()]
            if ids_to_keep:
                placeholders = ",".join("?" * len(ids_to_keep))
                c.execute(
                    f"DELETE FROM recently_viewed WHERE user_id=? AND id NOT IN ({placeholders})",
                    (user_id, *ids_to_keep),
                )

    def get_recently_viewed(self, user_id: int, limit: int = 8):
        with self._conn() as c:
            rows = c.execute(
                "SELECT product_json FROM recently_viewed WHERE user_id=? "
                "ORDER BY viewed_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [json.loads(r["product_json"]) for r in rows]

    # ─────────────────────── Admin Dashboard ───────────────────────
    def admin_summary(self) -> dict:
        """Platform-wide stats across ALL users, for the admin dashboard."""
        with self._conn() as c:
            total_users = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
            total_orders = c.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
            total_revenue = c.execute("SELECT COALESCE(SUM(total),0) AS s FROM orders").fetchone()["s"]
            total_reviews = c.execute("SELECT COUNT(*) AS n FROM reviews").fetchone()["n"]
            active_alerts = c.execute(
                "SELECT COUNT(*) AS n FROM price_alerts WHERE triggered=0"
            ).fetchone()["n"]
        return {
            "total_users": total_users,
            "total_orders": total_orders,
            "total_revenue": round(total_revenue, 2),
            "total_reviews": total_reviews,
            "active_alerts": active_alerts,
        }

    def admin_all_orders(self, limit: int = 500):
        """All orders, across all users, newest first — for admin reporting."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT o.*, u.username FROM orders o JOIN users u ON u.id = o.user_id "
                "ORDER BY o.placed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["items"] = json.loads(r["items_json"])
            out.append(d)
        return out

    def admin_signup_counts_by_day(self):
        with self._conn() as c:
            rows = c.execute(
                "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n "
                "FROM users GROUP BY day ORDER BY day"
            ).fetchall()
        return [{"day": r["day"], "count": r["n"]} for r in rows]

    # ───────────────────────── Trends (real usage signals) ─────────────────────────
    def get_most_viewed_products(self, limit: int = 8):
        """
        Real trending signal: aggregates the recently_viewed log across
        ALL users (not just one), so 'trending now' reflects actual
        traffic on the catalog rather than a simulated number.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT product_id, product_json, COUNT(*) AS views "
                "FROM recently_viewed GROUP BY product_id "
                "ORDER BY views DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            p = json.loads(r["product_json"])
            p["view_count"] = r["views"]
            out.append(p)
        return out

    def get_best_selling_products(self, limit: int = 8):
        """Real signal from actual placed orders — aggregates item
        quantities across every order in the system."""
        with self._conn() as c:
            rows = c.execute("SELECT items_json FROM orders").fetchall()
        counts = {}
        for r in rows:
            for item in json.loads(r["items_json"]):
                pid = str(item.get("product_id"))
                if pid not in counts:
                    counts[pid] = {"product": item, "count": 0}
                counts[pid]["count"] += 1
        ranked = sorted(counts.values(), key=lambda x: x["count"], reverse=True)
        return ranked[:limit]
