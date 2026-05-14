from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from humetric_embed import (
    CheckpointMalformed,
    CheckpointMissing,
    load_graph_embeddings,
    load_text_embeddings,
    load_tower_embeddings,
)


def _write_bundle(d: Path, kind: str, ids: list[str], vecs: np.ndarray) -> None:
    np.save(d / f"{kind}_vecs.npy", vecs)
    (d / f"{kind}_ids.json").write_text(json.dumps(ids))


def test_graph_roundtrip(tmp_path: Path) -> None:
    ids = ["gh:a", "gh:b", "gh:c"]
    vecs = np.random.randn(3, 8).astype(np.float32)
    _write_bundle(tmp_path, "graph", ids, vecs)

    bundle = load_graph_embeddings(tmp_path).unwrap()
    assert bundle.person_ids == tuple(ids)
    assert bundle.vectors.shape == (3, 8)
    assert bundle.dim == 8


def test_tower_and_text_loaders(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "tower", ["x"], np.ones((1, 4), dtype=np.float32))
    _write_bundle(tmp_path, "text", ["x"], np.ones((1, 16), dtype=np.float32))

    assert load_tower_embeddings(tmp_path).unwrap().dim == 4
    assert load_text_embeddings(tmp_path).unwrap().dim == 16


def test_missing_dir_returns_err(tmp_path: Path) -> None:
    r = load_graph_embeddings(tmp_path / "nope")
    assert r.is_err()
    assert isinstance(r.err(), CheckpointMissing)


def test_missing_ids_returns_err(tmp_path: Path) -> None:
    np.save(tmp_path / "graph_vecs.npy", np.zeros((1, 4), dtype=np.float32))
    r = load_graph_embeddings(tmp_path)
    assert r.is_err()
    assert isinstance(r.err(), CheckpointMissing)


def test_row_id_mismatch_is_malformed(tmp_path: Path) -> None:
    np.save(tmp_path / "graph_vecs.npy", np.zeros((3, 4), dtype=np.float32))
    (tmp_path / "graph_ids.json").write_text(json.dumps(["only-one"]))
    r = load_graph_embeddings(tmp_path)
    assert r.is_err()
    assert isinstance(r.err(), CheckpointMalformed)
