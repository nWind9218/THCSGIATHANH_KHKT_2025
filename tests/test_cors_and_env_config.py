from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from api.cors_config import get_cors_settings, is_origin_allowed


REQUIRED_ENV_KEYS = [
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "ALLOWED_ORIGINS",
    "ALLOWED_CREDENTIALS",
    "ALLOWED_METHODS",
    "ALLOWED_HEADERS",
    "CORS_MAX_AGE",
    "WS_REQUIRE_ORIGIN",
]


def _load_project_env() -> dict[str, str]:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    return {k: v for k, v in dotenv_values(env_path).items() if isinstance(v, str)}


def _build_cors_test_client(settings: dict) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings["allow_origins"],
        allow_origin_regex=settings["allow_origin_regex"],
        allow_credentials=settings["allow_credentials"],
        allow_methods=settings["allow_methods"],
        allow_headers=settings["allow_headers"],
        expose_headers=settings["expose_headers"],
        max_age=settings["max_age"],
    )

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


def test_env_required_keys_exist_and_not_empty() -> None:
    env = _load_project_env()
    missing = [k for k in REQUIRED_ENV_KEYS if not env.get(k, "").strip()]
    assert not missing, f"Missing or empty required env keys: {missing}"


def test_env_basic_value_formats() -> None:
    env = _load_project_env()

    assert env["DATABASE_URL"].startswith("postgresql://")
    assert env["REDIS_URL"].startswith(("redis://", "rediss://"))
    assert env["OPENAI_API_KEY"].startswith("sk-")

    assert env["ALLOWED_CREDENTIALS"].lower() in {"true", "false"}
    assert env["WS_REQUIRE_ORIGIN"].lower() in {"true", "false"}


def test_allowed_origins_are_normalized_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://mimi-hello.vercel.app/")
    settings = get_cors_settings()
    assert settings["allow_origins"] == ["https://mimi-hello.vercel.app"]


def test_http_cors_preflight_allows_configured_origin(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://mimi-hello.vercel.app/")
    monkeypatch.setenv("ALLOWED_CREDENTIALS", "true")
    monkeypatch.setenv("ALLOWED_METHODS", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
    monkeypatch.setenv("ALLOWED_HEADERS", "*")

    settings = get_cors_settings()
    client = _build_cors_test_client(settings)

    response = client.options(
        "/ping",
        headers={
            "Origin": "https://mimi-hello.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://mimi-hello.vercel.app"


def test_websocket_origin_validation_matches_allowed_origin(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://mimi-hello.vercel.app/")
    monkeypatch.setenv("WS_REQUIRE_ORIGIN", "true")

    settings = get_cors_settings()

    assert is_origin_allowed("https://mimi-hello.vercel.app", settings)
    assert not is_origin_allowed("https://evil.example.com", settings)


def test_websocket_origin_validation_allows_missing_origin_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://mimi-hello.vercel.app")
    monkeypatch.setenv("WS_REQUIRE_ORIGIN", "false")

    settings = get_cors_settings()

    assert is_origin_allowed(None, settings)
