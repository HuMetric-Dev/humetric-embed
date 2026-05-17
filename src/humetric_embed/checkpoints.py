from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from humetric_core import Err, Ok, Result

from humetric_embed.errors import CheckpointMalformed, CheckpointMissing, EmbedError


@dataclass(frozen=True, slots=True)
class EmbeddingBundle:
    """A `.npy` matrix paired with a list of entity ids in matching row order.

    Entity ids are the prefixed form (`p:gh:octocat`, `o:gh:anthropic`) so the
    same bundle can mix persons and organizations — the type is implicit in
    the id prefix and recoverable via `humetric_core.entity_type_of`.

    Produced by `humetric-train` jobs (LightGCN, two-tower, embed_corpus).
    """

    entity_ids: tuple[str, ...]
    vectors: np.ndarray  # shape (n, dim), float32

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


def _load_bundle(dir_path: Path, kind: str) -> Result[EmbeddingBundle, EmbedError]:
    if not dir_path.exists():
        return Err(CheckpointMissing(path=str(dir_path), expected=f"{kind} dir"))

    npy = dir_path / f"{kind}_vecs.npy"
    ids = dir_path / f"{kind}_ids.json"

    if not npy.exists():
        return Err(CheckpointMissing(path=str(npy), expected=f"{kind}_vecs.npy"))
    if not ids.exists():
        return Err(CheckpointMissing(path=str(ids), expected=f"{kind}_ids.json"))

    try:
        vecs = np.load(npy)
    except (OSError, ValueError) as e:
        return Err(CheckpointMalformed(path=str(npy), reason=str(e)))

    try:
        with open(ids, encoding="utf-8") as fh:
            ids_data = json.load(fh)
    except (OSError, ValueError) as e:
        return Err(CheckpointMalformed(path=str(ids), reason=str(e)))

    if not isinstance(ids_data, list) or not all(isinstance(x, str) for x in ids_data):
        return Err(CheckpointMalformed(path=str(ids), reason="ids must be a list[str]"))

    if vecs.ndim != 2:
        return Err(CheckpointMalformed(path=str(npy), reason=f"expected 2D, got {vecs.ndim}D"))
    if vecs.shape[0] != len(ids_data):
        return Err(
            CheckpointMalformed(
                path=str(dir_path),
                reason=f"rows={vecs.shape[0]} but ids={len(ids_data)}",
            )
        )

    return Ok(
        EmbeddingBundle(
            entity_ids=tuple(ids_data),
            vectors=np.ascontiguousarray(vecs, dtype=np.float32),
        )
    )


def load_graph_embeddings(dir_path: str | Path) -> Result[EmbeddingBundle, EmbedError]:
    """Read `<dir>/graph_vecs.npy` + `<dir>/graph_ids.json` written by train_lightgcn.py."""
    return _load_bundle(Path(dir_path), "graph")


def load_tower_embeddings(dir_path: str | Path) -> Result[EmbeddingBundle, EmbedError]:
    """Read `<dir>/tower_vecs.npy` + `<dir>/tower_ids.json` written by train_two_tower.py."""
    return _load_bundle(Path(dir_path), "tower")


def load_text_embeddings(dir_path: str | Path) -> Result[EmbeddingBundle, EmbedError]:
    """Read `<dir>/text_vecs.npy` + `<dir>/text_ids.json` written by embed_corpus.py."""
    return _load_bundle(Path(dir_path), "text")
