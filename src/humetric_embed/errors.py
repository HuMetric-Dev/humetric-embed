from __future__ import annotations

from dataclasses import dataclass

from humetric_core import HumetricError


@dataclass(frozen=True, slots=True)
class ModelLoadFailed(HumetricError):
    model: str
    reason: str


@dataclass(frozen=True, slots=True)
class EncodeFailed(HumetricError):
    n_inputs: int
    reason: str


@dataclass(frozen=True, slots=True)
class CheckpointMissing(HumetricError):
    path: str
    expected: str


@dataclass(frozen=True, slots=True)
class CheckpointMalformed(HumetricError):
    path: str
    reason: str


type EmbedError = ModelLoadFailed | EncodeFailed | CheckpointMissing | CheckpointMalformed
