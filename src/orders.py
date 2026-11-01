"""
orders.py — Demo order status simulation, coupon codes, and OTP.
No real payment, SMS, or shipping integration (this is a project demo) —
status progresses automatically based on time elapsed since the order
was placed, so a reviewer can watch it change live without waiting days.
"""

import random
from datetime import datetime, timedelta

COUPONS = {
    "WELCOME10": 10,
    "SHOP20": 20,
    "MCA50": 50,
}

PAYMENT_METHODS = {
    "cod": "💵 Cash on Delivery",
    "upi": "📱 UPI Payment",
    "cash": "💴 Cash (Pay at Pickup)",
}


def generate_upi_txn_id() -> str:
    """Demo UPI transaction reference — there's no real payment gateway
    here, so this simulates the reference number a real UPI app (GPay,
    PhonePe, Paytm) would show after a successful payment."""
    return f"UPI{random.randint(10**11, 10**12 - 1)}"

# (minutes_elapsed_threshold, label, icon)
TRACKING_STAGES = [
    (0, "Order Placed", "🛒"),
    (1, "Packed", "📦"),
    (3, "Shipped", "🚚"),
    (6, "Out for Delivery", "🛵"),
    (9, "Delivered", "✅"),
]


def apply_coupon(subtotal: float, code: str):
    """Returns (discount_amount, message)."""
    code = (code or "").strip().upper()
    if not code:
        return 0.0, None
    pct = COUPONS.get(code)
    if pct is None:
        return 0.0, f"'{code}' is not a valid coupon code."
    discount = round(subtotal * pct / 100, 2)
    return discount, f"Coupon '{code}' applied — {pct}% off (₹{discount:.0f})."


def get_tracking_info(placed_at_iso: str) -> dict:
    """
    Returns the full tracking picture for an order: which stage it's
    currently at, progress percentage (for a progress bar), and an
    estimated delivery date — everything the Orders page needs to draw
    a real-feeling tracker without any actual logistics integration.
    """
    placed_at = datetime.fromisoformat(placed_at_iso)
    elapsed_minutes = (datetime.utcnow() - placed_at).total_seconds() / 60

    current_index = 0
    for i, (minutes, _, _) in enumerate(TRACKING_STAGES):
        if elapsed_minutes >= minutes:
            current_index = i

    total_stages = len(TRACKING_STAGES)
    percent = int(round((current_index / (total_stages - 1)) * 100))
    label = TRACKING_STAGES[current_index][1]
    icon = TRACKING_STAGES[current_index][2]

    est_delivery = placed_at + timedelta(days=4)

    return {
        "current_index": current_index,
        "current_label": label,
        "current_icon": icon,
        "percent": percent,
        "stages": TRACKING_STAGES,
        "estimated_delivery": est_delivery.strftime("%d %b %Y"),
        "is_delivered": current_index == total_stages - 1,
    }


def get_order_status(placed_at_iso: str) -> str:
    """Kept for backward compatibility — short text status only."""
    info = get_tracking_info(placed_at_iso)
    return f"{info['current_icon']} {info['current_label']}"


def generate_otp() -> str:
    """6-digit demo OTP. There's no real SMS gateway here, so the OTP is
    shown directly in the UI — clearly labeled as a demo — instead of
    actually being sent anywhere."""
    return f"{random.randint(0, 999999):06d}"
