"""Authentication for the teacher console: pbkdf2 password hashes, HMAC-signed session cookie, login rate limit.

Everything is stdlib-only. Configuration is read from the environment on every call (never at import time).
"""

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

COOKIE_NAME = "mimi_console"
COOKIE_PATH = "/console"
SESSION_TTL = 12 * 3600
HASH_ALGORITHM = "pbkdf2_sha256"
MIN_ITERATIONS = 390_000
MIN_SECRET_LENGTH = 32
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "mimi-console"
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 300

DISABLED_MESSAGE = "Bàn hỗ trợ chưa được cấu hình"
UNAUTHORIZED_MESSAGE = "Vui lòng đăng nhập"
CSRF_MESSAGE = "Yêu cầu không hợp lệ"
RATE_LIMIT_MESSAGE = "Đăng nhập sai quá nhiều lần, vui lòng thử lại sau ít phút"
BAD_LOGIN_MESSAGE = "Sai tài khoản hoặc mật khẩu"


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _urlsafe_e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _urlsafe_d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str, *, iterations: int = MIN_ITERATIONS, salt: bytes | None = None) -> str:
    """Return `pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>`."""
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{HASH_ALGORITHM}${iterations}${_b64e(salt)}${_b64e(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of `password` against a stored hash. Malformed hashes never match."""
    try:
        algorithm, iterations, salt_b64, hash_b64 = stored.strip().split("$")
        if algorithm != HASH_ALGORITHM:
            return False
        rounds = int(iterations)
        if rounds < MIN_ITERATIONS:
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(hash_b64, validate=True)
    except (ValueError, binascii.Error):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds, dklen=len(expected))
    return hmac.compare_digest(actual, expected)


def sign_session(username: str, secret: str, *, ttl: int = SESSION_TTL, now: float | None = None) -> str:
    """Build the cookie value: base64url(json{"u","exp"}) + "." + hmac_sha256_hex."""
    issued = time.time() if now is None else now
    payload = _urlsafe_e(json.dumps({"u": username, "exp": int(issued + ttl)}, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def read_session(token: str | None, secret: str, *, now: float | None = None) -> str | None:
    """Return the username of a valid, unexpired session cookie, else None."""
    if not token or "." not in token:
        return None
    payload, _, signature = token.rpartition(".")
    expected = hmac.new(secret.encode("utf-8"), payload.encode("ascii", "ignore"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        data = json.loads(_urlsafe_d(payload))
        username, expires = data["u"], float(data["exp"])
    except (ValueError, KeyError, TypeError, binascii.Error):
        return None
    if not isinstance(username, str) or not username:
        return None
    return username if (time.time() if now is None else now) < expires else None


@dataclass(frozen=True)
class ConsoleConfig:
    username: str
    password_hash: str
    secret: str
    cookie_secure: bool


def load_config() -> ConsoleConfig | None:
    """Current console configuration, or None (console disabled) when anything is missing."""
    username = os.getenv("CONSOLE_ADMIN_USERNAME", "").strip()
    password_hash = os.getenv("CONSOLE_ADMIN_PASSWORD_HASH", "").strip()
    secret = os.getenv("CONSOLE_SESSION_SECRET", "")
    if not username or not password_hash or len(secret) < MIN_SECRET_LENGTH:
        return None
    secure = os.getenv("CONSOLE_COOKIE_SECURE", "true").strip().lower() != "false"
    return ConsoleConfig(username, password_hash, secret, secure)


def check_credentials(config: ConsoleConfig, username: str, password: str) -> bool:
    """Verify both fields without short-circuiting, so timing does not reveal which was wrong."""
    user_ok = hmac.compare_digest(username.encode("utf-8"), config.username.encode("utf-8"))
    password_ok = verify_password(password, config.password_hash)
    return user_ok and password_ok


class LoginRateLimiter:
    """In-memory failed-login limiter: `max_failures` per key per `window` seconds."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, window: float = LOGIN_WINDOW_SECONDS):
        self.max_failures, self.window = max_failures, window
        self._failures: dict[str, list[float]] = {}

    def _recent(self, key: str, now: float) -> list[float]:
        recent = [stamp for stamp in self._failures.get(key, []) if now - stamp < self.window]
        if recent:
            self._failures[key] = recent
        else:
            self._failures.pop(key, None)
        return recent

    def blocked(self, key: str, now: float | None = None) -> bool:
        return len(self._recent(key, time.time() if now is None else now)) >= self.max_failures

    def record_failure(self, key: str, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        self._recent(key, moment)
        self._failures.setdefault(key, []).append(moment)
        if len(self._failures) > 10_000:
            for stale in [k for k in self._failures if not self._recent(k, moment)]:
                self._failures.pop(stale, None)

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._failures.clear()
        else:
            self._failures.pop(key, None)


login_limiter = LoginRateLimiter()


def client_ip(request: Request) -> str:
    """Caller address; honours the last X-Forwarded-For hop only when CONSOLE_TRUST_PROXY=true."""
    if os.getenv("CONSOLE_TRUST_PROXY", "false").strip().lower() == "true":
        forwarded = request.headers.get("x-forwarded-for", "")
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"


def guard(request: Request) -> ConsoleConfig:
    """Dependency: console must be configured (503) and mutating requests must carry the CSRF header (403)."""
    config = load_config()
    if config is None:
        raise HTTPException(status_code=503, detail=DISABLED_MESSAGE)
    if request.method in MUTATING_METHODS and request.headers.get(CSRF_HEADER, "") != CSRF_VALUE:
        raise HTTPException(status_code=403, detail=CSRF_MESSAGE)
    return config


def current_user(request: Request) -> str:
    """Dependency: guard + a valid session cookie; returns the username."""
    config = guard(request)
    username = read_session(request.cookies.get(COOKIE_NAME), config.secret)
    if username is None or not hmac.compare_digest(username.encode("utf-8"), config.username.encode("utf-8")):
        raise HTTPException(status_code=401, detail=UNAUTHORIZED_MESSAGE)
    return username
