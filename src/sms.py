"""
sms.py — Real SMS OTP delivery via Twilio, with automatic demo fallback.

Sending a real SMS requires YOUR OWN Twilio account (https://twilio.com):
  1. Sign up, get a free trial (comes with a small credit).
  2. Get a Twilio phone number (Console → Phone Numbers).
  3. Copy your Account SID + Auth Token (Console dashboard).
  4. On a TRIAL account, you must "verify" the destination number first
     (Console → Phone Numbers → Verified Caller IDs) — Twilio trial
     accounts can only text numbers you've explicitly verified.
  5. Put these three values in your .env file:
       TWILIO_ACCOUNT_SID=...
       TWILIO_AUTH_TOKEN=...
       TWILIO_FROM_NUMBER=+1xxxxxxxxxx

If these are not set (or sending fails for any reason — unverified
number, no credit, etc.), this module returns a clear failure reason
and the app falls back to showing the OTP on screen (demo mode) —
it never crashes the checkout flow either way.
"""

import os


def sms_configured() -> bool:
    return bool(
        os.getenv("TWILIO_ACCOUNT_SID")
        and os.getenv("TWILIO_AUTH_TOKEN")
        and os.getenv("TWILIO_FROM_NUMBER")
    )


def send_otp_sms(phone_number_10digit: str, otp: str, country_code: str = "+91"):
    """
    Attempts to send a real SMS via Twilio.
    Returns (success: bool, message: str).
    """
    if not sms_configured():
        return False, "SMS not configured (no Twilio credentials in .env) — using demo mode."

    try:
        from twilio.rest import Client
    except ImportError:
        return False, "The 'twilio' package isn't installed — run: pip install twilio"

    try:
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        to_number = f"{country_code}{phone_number_10digit}"
        client.messages.create(
            body=f"Your ShopAI verification code is {otp}. It expires in 5 minutes.",
            from_=os.getenv("TWILIO_FROM_NUMBER"),
            to=to_number,
        )
        return True, f"OTP sent to {to_number}."
    except Exception as e:
        return False, f"Could not send real SMS ({e}) — falling back to demo mode."
