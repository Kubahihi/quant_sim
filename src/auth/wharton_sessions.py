"""Revocable Wharton sessions, independent of personal user IDs."""
import hashlib
import hmac
import math
import time
from collections.abc import Mapping

ABSOLUTE_SESSION_SECONDS = 8 * 60 * 60
IDLE_SESSION_SECONDS = 30 * 60


def _fingerprint(user: Mapping, password_hash: str) -> str:
    material = "\0".join(str(user.get(key, "")) for key in ("id", "username", "role", "primary_module"))
    return hashlib.sha256((material + "\0" + password_hash).encode("utf-8")).hexdigest()


def issue_session(user: Mapping, password_hash: str, *, now: float | None = None) -> dict:
    timestamp = time.time() if now is None else float(now)
    return {**dict(user), "_session_issued_at": timestamp, "_session_last_seen": timestamp,
            "_session_fingerprint": _fingerprint(user, password_hash)}


def session_is_current(profile: Mapping, user: Mapping, password_hash: str, *, now: float | None = None) -> bool:
    timestamp = time.time() if now is None else float(now)
    try:
        issued, seen = float(profile["_session_issued_at"]), float(profile["_session_last_seen"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(x) for x in (issued, seen, timestamp)):
        return False
    if not issued <= seen <= timestamp or timestamp - issued >= ABSOLUTE_SESSION_SECONDS or timestamp - seen >= IDLE_SESSION_SECONDS:
        return False
    fingerprint = profile.get("_session_fingerprint")
    return isinstance(fingerprint, str) and hmac.compare_digest(fingerprint, _fingerprint(user, password_hash))
