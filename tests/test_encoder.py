from __future__ import annotations

from humetric_embed import ModelLoadFailed, TextEncoder


def test_lazy_construction_does_not_load() -> None:
    enc = TextEncoder()
    # Before .load() the model is unloaded and encode_batch errors cleanly.
    r = enc.encode_batch(["hello"])
    assert r.is_err()
    err = r.err()
    assert isinstance(err, ModelLoadFailed)
    assert "not loaded" in err.reason


def test_model_id_alias_resolution() -> None:
    assert TextEncoder(model="bge-small").model_id == "BAAI/bge-small-en-v1.5"
    assert TextEncoder(model="bge-large").model_id == "BAAI/bge-large-en-v1.5"
    # Unknown alias passes through as a raw model id.
    assert TextEncoder(model="foo/bar").model_id == "foo/bar"


def test_empty_input_after_load_is_zero_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Don't load a real model in CI; simulate the loaded state minimally.
    enc = TextEncoder()
    # Inject a fake model that supports the encode interface we use.

    class _Fake:
        def encode(self, texts, **_kw):  # type: ignore[no-untyped-def]
            import numpy as np

            return np.zeros((len(texts), 8), dtype=np.float32)

        def get_sentence_embedding_dimension(self) -> int:
            return 8

    enc._model = _Fake()  # type: ignore[assignment]
    enc._dim = 8
    r = enc.encode_batch([])
    assert r.unwrap().shape == (0, 8)
    r2 = enc.encode_batch(["a", "b"])
    assert r2.unwrap().shape == (2, 8)
