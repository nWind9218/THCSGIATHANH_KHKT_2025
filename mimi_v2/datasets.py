"""Validated conversation datasets; fixtures are not acceptance evidence."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
EXPECTATIONS = {"contains", "not_contains", "intent", "role", "support_mode", "handoff", "max_questions", "max_chars"}
SETUP_FIELDS = {"messages", "known", "user_role", "topic", "current_topic", "handoff_status", "notification_sent"}


def validate_dataset(data: dict) -> dict:
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Dataset must be an object with version=1")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for case in data["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not ID.fullmatch(case["id"]):
            raise ValueError("Each case requires a safe unique id")
        if case["id"] in seen:
            raise ValueError("Duplicate case id")
        seen.add(case["id"])
        setup = case.get("setup", {})
        if not isinstance(setup, dict) or set(setup) - SETUP_FIELDS:
            raise ValueError("Unknown local setup field")
        if "messages" in setup:
            if not isinstance(setup["messages"], list) or any(
                not isinstance(m, dict) or set(m) != {"role", "content"}
                or m["role"] not in {"user", "assistant"} or not isinstance(m["content"], str)
                for m in setup["messages"]
            ):
                raise ValueError("Setup messages require role/content")
        if "known" in setup and (not isinstance(setup["known"], dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in setup["known"].items()
        )):
            raise ValueError("Setup known must map strings to strings")
        if setup.get("handoff_status", "none") not in {"none", "pending", "accepted"}:
            raise ValueError("Unknown setup handoff status")
        if setup.get("user_role", "unknown") not in {"unknown", "student", "parent"}:
            raise ValueError("Unknown setup role")
        if "notification_sent" in setup and type(setup["notification_sent"]) is not bool:
            raise ValueError("Setup notification_sent must be boolean")
        if any(k in setup and not isinstance(setup[k], str) for k in ("topic", "current_topic")):
            raise ValueError("Setup topic must be a string")
        knowledge = case.get("knowledge", {})
        if not isinstance(knowledge, dict) or set(knowledge) - {"psychology", "student_knowledge", "user_memory"} or any(not isinstance(v, str) for v in knowledge.values()):
            raise ValueError("Knowledge must be local string fixtures")
        if case.get("role", "unknown") not in {"student", "parent", "unknown"}:
            raise ValueError("role must be student, parent or unknown")
        if not isinstance(case.get("turns"), list) or not case["turns"]:
            raise ValueError("Each case requires nonempty turns")
        for turn in case["turns"]:
            if not isinstance(turn, dict) or not isinstance(turn.get("user"), str) or not turn["user"].strip():
                raise ValueError("Each turn requires nonempty user text")
            if turn.get("action", "send") not in {"send", "acknowledge", "resume", "reset"}:
                raise ValueError("Unknown local turn action")
            if turn.get("fault", "none") not in {"none", "model", "store", "notify"}:
                raise ValueError("Unknown local fault")
            expected = turn.get("expect", {})
            if not isinstance(expected, dict) or set(expected) - EXPECTATIONS:
                raise ValueError("Unknown expectation; allowed: " + ", ".join(sorted(EXPECTATIONS)))
            for field in ("contains", "not_contains"):
                if field in expected and (not isinstance(expected[field], list) or not all(isinstance(x, str) and x for x in expected[field])):
                    raise ValueError(field + " must be a list of nonempty strings")
            for field in ("max_questions", "max_chars"):
                if field in expected and (type(expected[field]) is not int or expected[field] < 0):
                    raise ValueError(field + " must be a nonnegative integer")
            for field in ("intent", "role", "support_mode", "handoff"):
                if field in expected and not isinstance(expected[field], str):
                    raise ValueError(field + " must be a string")
            review = turn.get("review", [])
            if not isinstance(review, list) or not all(isinstance(x, str) and x.strip() for x in review):
                raise ValueError("review must contain human-review criteria strings")
    return data


def load_dataset(path: str | Path) -> dict:
    return validate_dataset(json.loads(Path(path).read_text(encoding="utf-8")))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
