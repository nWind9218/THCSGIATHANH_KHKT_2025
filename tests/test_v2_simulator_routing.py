from mimi_v2 import simulator


def test_routing_follows_env_and_flag_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MIMI_V2_ROUTING", raising=False)
    assert simulator.resolve_routing() == "off"
    monkeypatch.setenv("MIMI_V2_ROUTING", "Adaptive")
    assert simulator.resolve_routing() == "adaptive"
    assert simulator.resolve_routing("off") == "off"
    monkeypatch.delenv("MIMI_V2_ROUTING")
    assert simulator.resolve_routing("adaptive") == "adaptive"


def test_local_env_reads_routing_keys_from_dotenv(monkeypatch, tmp_path):
    for key in ("MIMI_V2_ROUTING", "MIMI_V2_MODEL_ASSESS", "FACEBOOK_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("MIMI_V2_ROUTING=adaptive\nMIMI_V2_MODEL_ASSESS=m\nFACEBOOK_TOKEN=x\n", encoding="utf-8")
    loaded = simulator.load_local_model_environment(env)
    assert loaded["MIMI_V2_ROUTING"] == "adaptive" and loaded["MIMI_V2_MODEL_ASSESS"] == "m"
    assert "FACEBOOK_TOKEN" not in loaded
