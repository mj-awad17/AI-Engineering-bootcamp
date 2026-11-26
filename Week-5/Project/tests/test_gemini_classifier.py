from types import SimpleNamespace

import pytest
from google.genai.errors import APIError

from app.services import gemini_classifier


def test_live_mode_uses_documented_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    calls = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["model"] = model
            calls["config"] = config
            return SimpleNamespace(
                text=(
                    '{"category":"Technical Support","confidence":0.8,'
                    '"reasoning":"Test response."}'
                )
            )

    class FakeClient:
        models = FakeModels()

        def __init__(self, *, api_key):
            assert api_key == "test-key"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(gemini_classifier.genai, "Client", FakeClient)

    prediction = gemini_classifier.classify_with_gemini("The app crashes.")

    assert calls["model"] == "gemini-3.5-flash-lite"
    assert not {"temperature", "top_p", "top_k", "candidate_count"} & calls["config"].keys()
    assert prediction.model == "gemini-3.5-flash-lite"


def test_model_not_found_error_identifies_model_setting(monkeypatch):
    monkeypatch.setenv("GEMINI_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "unsupported-model")

    class FakeModels:
        def generate_content(self, **kwargs):
            raise APIError(404, {"error": {"message": "model not found"}})

    class FakeClient:
        models = FakeModels()

        def __init__(self, *, api_key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(gemini_classifier.genai, "Client", FakeClient)

    with pytest.raises(gemini_classifier.GeminiClassificationError) as exc_info:
        gemini_classifier.classify_with_gemini("The app crashes.")

    assert "unsupported-model" in str(exc_info.value)
    assert "GEMINI_MODEL" in str(exc_info.value)
