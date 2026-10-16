"""CORS and Origin validation helpers for HTTP and WebSocket."""

from __future__ import annotations

import os
import re


def parse_csv_env(value: str | None, default: list[str] | None = None) -> list[str]:
    if not value:
        return default[:] if default else []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_origin(origin: str) -> str:
    """Normalize origin format by trimming whitespace and trailing slash."""
    return origin.strip().rstrip("/")


def parse_bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_env(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def get_cors_settings() -> dict:
    raw_allow_origins = parse_csv_env(os.getenv("ALLOWED_ORIGINS"), ["*"])
    allow_origins = [
        "*" if origin == "*" else normalize_origin(origin)
        for origin in raw_allow_origins
    ]
    allow_methods = parse_csv_env(
        os.getenv("ALLOWED_METHODS"),
        ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    allow_headers = parse_csv_env(os.getenv("ALLOWED_HEADERS"), ["*"])
    expose_headers = parse_csv_env(os.getenv("EXPOSE_HEADERS"), [])

    return {
        "allow_origins": allow_origins,
        "allow_origin_regex": os.getenv("ALLOWED_ORIGIN_REGEX") or None,
        "allow_methods": allow_methods,
        "allow_headers": allow_headers,
        "expose_headers": expose_headers,
        "allow_credentials": parse_bool_env(os.getenv("ALLOWED_CREDENTIALS"), True),
        "max_age": parse_int_env(os.getenv("CORS_MAX_AGE"), 600),
        "ws_require_origin": parse_bool_env(os.getenv("WS_REQUIRE_ORIGIN"), False),
    }


def is_origin_allowed(origin: str | None, settings: dict) -> bool:
    # Non-browser clients may not send Origin. Allow by default unless forced.
    if not origin:
        return not settings.get("ws_require_origin", False)

    normalized_origin = normalize_origin(origin)
    allow_origins: list[str] = settings.get("allow_origins", [])
    if "*" in allow_origins:
        return True
    if normalized_origin in allow_origins:
        return True

    pattern = settings.get("allow_origin_regex")
    if pattern and re.match(pattern, normalized_origin):
        return True

    return False
