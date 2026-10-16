"""Tests deny network, production imports and credential-file reads."""
from copy import deepcopy
import builtins
import io
import json
import os
from pathlib import Path

import pytest

from mimi_eval.contracts import ROOT, fingerprint
from mimi_eval.replay import network_denied


@pytest.fixture(autouse=True)
def offline_guard(monkeypatch):
    original_import = builtins.__import__
    forbidden = {"openai", "supabase", "redis", "dotenv", "mimi_v2", "api", "memory"}
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError("production_or_provider_import_denied")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for module in (builtins, io):
        original = module.open
        def guarded_open(file, *args, _original=original, **kwargs):
            if isinstance(file, (str, os.PathLike)) and Path(file).name.startswith(".env"):
                raise AssertionError("credential_file_read_denied")
            return _original(file, *args, **kwargs)
        monkeypatch.setattr(module, "open", guarded_open)
    original_getenv = os.getenv
    def guarded_getenv(key, *args):
        if key.startswith(("OPENAI_", "SUPABASE_", "MIMI_", "REDIS_")):
            raise AssertionError("production_environment_read_denied")
        return original_getenv(key, *args)
    monkeypatch.setattr(os, "getenv", guarded_getenv)
    with network_denied():
        yield


@pytest.fixture
def suite():
    return json.loads((ROOT / "fixtures" / "mimi-contrastive-suite-v0.2.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample(suite):
    row = deepcopy(next(x for x in suite["cases"] if x["fixture_id"] == "T01-C1"))
    return row["input"], row["scripted_observation"]


def rebind(case, observation):
    observation["input_sha256"] = fingerprint(case)


def make_ref(source):
    return {"message_id": source["message_id"], "source_sha256": source["sha256"],
            "role": source["role"], "quote": source["text"], "start": 0, "end": len(source["text"])}
