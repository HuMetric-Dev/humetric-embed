# humetric-embed

Inference-side encoder wrappers for the Humetric workspace.

## Encoders

- `TextEncoder` — wraps `sentence-transformers`. Default model is `BAAI/bge-small-en-v1.5` (CPU-friendly). Set `model="bge-large"` to load `BAAI/bge-large-en-v1.5` (heavy; for the GPU box). Both are downloaded from HuggingFace on first use; you can also point at a local checkpoint dir.
- `load_graph_embeddings` — reads node-embedding `.npy` + ids manifest produced by `humetric-train` (LightGCN).
- `load_tower_embeddings` — same shape, for the two-tower person-tower output.

Everything returns `Result[..., EmbedError]`. No raised exceptions cross the package boundary.

## Usage

```python
from humetric_embed import TextEncoder

enc = TextEncoder().load().unwrap()
vecs = enc.encode_batch(["rust engineer", "kafka payments"]).unwrap()
# vecs.shape == (2, dim)
```
