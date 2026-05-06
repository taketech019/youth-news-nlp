import importlib
import pytest


def test_config_loads_nim_constants(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_NIM_BASE_URL", "https://test.api.com/v1")
    monkeypatch.setenv("NVIDIA_NIM_MODEL", "test-model")

    import rag_chatbot.config as cfg
    importlib.reload(cfg)

    assert cfg.NIM_API_KEY == "test-key"
    assert cfg.NIM_BASE_URL == "https://test.api.com/v1"
    assert cfg.NIM_MODEL == "test-model"


def test_config_defaults(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "k")
    import rag_chatbot.config as cfg
    importlib.reload(cfg)

    assert cfg.TOP_K == 5
    assert cfg.CHROMA_COLLECTION == "chunyeon_news"
    assert set(cfg.DATA_FILES.keys()) == {"동아일보", "한겨레", "한국일보"}
