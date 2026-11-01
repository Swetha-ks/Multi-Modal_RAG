"""
auth.py — Password hashing utilities (stdlib only)
────────────────────────────────────────────────────
Uses PBKDF2-HMAC-SHA256 with a random per-user salt. No third-party
crypto dependency is required, so this works out of the box on any
platform (Windows/Mac/Linux) without extra installs.
"""

import hashlib
import os
import binascii

PBKDF2_ITERATIONS = 260_000


def hash_password(password: str):
    """Returns (hash_hex, salt_hex) for a freshly chosen random salt."""
    salt = binascii.hexlify(os.urandom(16)).decode()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return binascii.hexlify(pwd_hash).decode(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return binascii.hexlify(pwd_hash).decode() == stored_hash


def validate_credentials(username: str, password: str):
    """Returns an error string, or None if valid."""
    if not username or len(username.strip()) < 3:
        return "Username must be at least 3 characters."
    if not password or len(password) < 6:
        return "Password must be at least 6 characters."
    return None
