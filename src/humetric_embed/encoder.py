from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from humetric_core import Err, Ok, Result

from humetric_embed.errors import EmbedError, EncodeFailed, ModelLoadFailed

_MODEL_ALIASES: dict[str, str] = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "mxbai-large": "mixedbread-ai/mxbai-embed-large-v1",
}


@dataclass(slots=True)
class TextEncoder:
    """Lazy sentence-transformers wrapper.

    Two-step construction so `__init__` is total and never touches the
    network: `TextEncoder(model=...)` is free, then `.load()` actually
    downloads/loads weights and returns a Result.
    """

    model: str = "bge-small"
    cache_dir: str | None = None
    _model: Any | None = field(default=None, init=False, repr=False)
    _dim: int = field(default=0, init=False)

    @property
    def model_id(self) -> str:
        return _MODEL_ALIASES.get(self.model, self.model)

    @property
    def dim(self) -> int:
        if self._model is None:
            return 0
        return self._dim

    def load(self) -> Result[TextEncoder, EmbedError]:
        if self._model is not None:
            return Ok(cast(TextEncoder, self))
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            return Err(ModelLoadFailed(model=self.model_id, reason=f"import: {e}"))

        try:
            m = SentenceTransformer(self.model_id, cache_folder=self.cache_dir)
        except (OSError, ValueError, RuntimeError) as e:
            return Err(ModelLoadFailed(model=self.model_id, reason=str(e)))

        self._model = m
        try:
            self._dim = int(m.get_sentence_embedding_dimension() or 0)
        except (AttributeError, RuntimeError):
            self._dim = 0
        return Ok(cast(TextEncoder, self))

    def encode_batch(self, texts: list[str]) -> Result[np.ndarray, EmbedError]:
        if self._model is None:
            return Err(ModelLoadFailed(model=self.model_id, reason="not loaded"))
        if not texts:
            return Ok(np.zeros((0, self._dim), dtype=np.float32))
        try:
            arr = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except (RuntimeError, ValueError) as e:
            return Err(EncodeFailed(n_inputs=len(texts), reason=str(e)))
        return Ok(np.asarray(arr, dtype=np.float32))

    def encode_one(self, text: str) -> Result[np.ndarray, EmbedError]:
        r = self.encode_batch([text])
        if isinstance(r, Err):
            return r
        return Ok(r.value[0])


def text_encoder_from_checkpoint(path: str | Path) -> Result[TextEncoder, EmbedError]:
    """Load a TextEncoder pointing at a local sentence-transformers checkpoint."""
    p = Path(path)
    if not p.exists():
        return Err(ModelLoadFailed(model=str(p), reason="path does not exist"))
    enc = TextEncoder(model=str(p))
    return enc.load()
