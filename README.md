# Gauss — GMM+ANS LLM Checkpoint Compression

> **4.90× smaller. CPU-only. Lossless format, bounded error. Full dtype round-trip.**

**Gauss** compresses large language model weight tensors using a two-stage pipeline:
Gaussian Mixture Model (GMM) distribution fitting followed by Asymmetric Numeral Systems
(ANS) entropy coding. Small or 1-D tensors (e.g. LayerNorm scale/bias) are stored
**losslessly** in a dedicated raw section — nothing is silently dropped.

```
1,645 MB safetensors  →  335 MB .gauss   (4.90×)   max error ±5×10⁻⁴
```

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python ≥ 3.9](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org)

---

## Highlights

| Feature | Detail |
|---|---|
| **Compression ratio** | ~4.9× on typical SFT checkpoints (K=16) |
| **Max reconstruction error** | ±5×10⁻⁴ for fp32/fp16; ±3.9×10⁻³ for bf16 (adaptive-S) |
| **Small / 1-D tensors** | Stored **losslessly** in raw section (not skipped) |
| **Dtype preservation** | Original `float32 / float16 / bfloat16` restored exactly |
| **Parallel** | Compression and decompression via `multiprocessing.Pool` |
| **Random access** | Decompress one tensor without reading the whole file |
| **CPU-only** | No GPU required |
| **Dependencies** | NumPy, PyTorch, safetensors, constriction |

---

## Installation

```bash
pip install gauss-compress
```

Or from source:

```bash
git clone https://github.com/skystarry-ai/gauss
cd gauss
pip install -e .
```

**Requirements:** Python ≥ 3.9, and the four packages above (installed automatically).

---

## Quick Start

```bash
# Compress a safetensors checkpoint
gauss compress model.safetensors model.gauss

# Restore it — original dtypes are recovered automatically
gauss decompress model.gauss restored.safetensors

# Inspect a compressed file (no decompression)
gauss info model.gauss
```

---

## What Gets Compressed vs. Stored Raw

Gauss automatically routes each tensor to the best storage path:

| Condition | Storage path | Lossless? |
|---|---|---|
| `ndim == 1` (e.g. bias, LayerNorm scale) | Raw section | ✅ Yes |
| `numel < 32 768` (small tensors) | Raw section | ✅ Yes |
| Everything else | GMM+ANS compressed | ❌ Bounded ±5×10⁻⁴ |

You can adjust the raw threshold with `--raw-threshold` (default: 32 768).

---

## CLI Reference

### `gauss compress`

```
gauss compress INPUT OUTPUT [options]
```

| Option | Default | Description |
|---|---|---|
| `--K` | `16` | GMM components per tensor |
| `--max-iter` | `100` | Max hard-EM iterations |
| `--workers` | all cores | Parallel worker processes |
| `--shard-threshold` | `5 000 000` | Split 2-D+ tensors larger than this into row-shards |
| `--shard-rows` | `4 096` | Rows per shard |
| `--raw-threshold` | `32 768` | Store tensors smaller than this losslessly |
| `--layers` | *(all)* | Comma-separated layer names to compress |

**Example output**

```
Loading sft_model.safetensors ...
Layers: 170 compressed (148 sharded), 42 raw | tasks: 892 | K=16 | workers=8
  [   1/170] model.layers.0.mlp.down_proj.weight        5.412x  iter= 23 [8 shards]  3.2s
  ...
  [raw] model.layers.0.self_attn.q_proj.bias
  ...
Saved: sft_model.gauss  (335.8 MB)
Overall ratio: 4.903x  (1645MB → 335MB)  total 355s
```

---

### `gauss decompress`

```
gauss decompress INPUT OUTPUT [--workers N]
```

Restores the original `.safetensors` file with all tensors in their **original dtype**
(`float32`, `float16`, `bfloat16`, …).

---

### `gauss info`

```
gauss info FILE
```

```
File:    model.gauss
Size:    335.84 MB
Layers:  212  (170 compressed, 42 raw)

  model.layers.0.mlp.gate_proj.weight            18432KB →  3913KB  4.709x  K=16  torch.float32
  model.layers.0.mlp.down_proj.weight            18432KB →  3413KB  5.400x  K=16  torch.float32 [×8 shards]
  ...

  model.layers.0.self_attn.q_proj.bias               3KB   [raw / lossless]  torch.float32
  model.norm.weight                                   3KB   [raw / lossless]  torch.float32
```

---

## Python API

```python
from gauss import GaussWriter, GaussReader, compress_file, decompress_file

# --- High-level API (recommended) ---
compress_file("model.safetensors", "model.gauss", K=16, n_workers=8)
decompress_file("model.gauss", "restored.safetensors")

# --- Low-level writer ---
writer = GaussWriter("model.gauss")

# Add a GMM-compressed layer
writer.add(key, shape, K, pi, mu, sigma, idx_compressed, resid_compressed,
           dtype_id=0)   # 0 = float32

# Add a raw (lossless) layer
writer.add_raw(key, shape, dtype_id=0, raw_bytes=tensor.numpy().tobytes())

writer.save()
print(writer.summary())

# --- Low-level reader ---
reader = GaussReader("model.gauss")
print(reader.keys())                        # all tensor names

arr = reader.decompress("model.layers.0.mlp.gate_proj.weight")  # np.ndarray
state_dict = reader.decompress_all()        # {key: np.ndarray}
```

---

## File Format (GAUSS004)

The `.gauss` format stores metadata and data in separate sections, enabling random
access to any individual tensor with a single seek:

```
magic[8]           "GAUSS004"
n_compressed[4]    uint32 — GMM-compressed layer count
n_raw[4]           uint32 — raw lossless layer count

=== Compressed index section ===
  per layer: name, shape, dtype_id, n_shards, shard_rows,
             per shard: K, pi, mu, sigma, resid_scale, idx_words, resid_words

=== Raw index section ===
  per layer: name, shape, dtype_id, raw_n_bytes

=== Compressed data section ===
  per shard: ANS index stream + ANS residual stream

=== Raw data section ===
  per layer: little-endian tensor bytes
```

`resid_scale` (uint16, new in GAUSS004) records the effective quantization scale
used per shard. For fp32 tensors this is always 1000; for bf16 it is automatically
capped at 128 via adaptive-S (see [How It Works](#how-it-works)).

Legacy `GAUSS001`–`GAUSS003` files are read-compatible (dtype defaults to float32,
resid_scale defaults to 1000).

---

## Module Layout

```
gauss/
├── __init__.py     # Public API: GaussWriter, GaussReader, compress_file, decompress_file
├── gmm.py          # Pure-NumPy hard EM: fit_gmm_fast, assign_all
├── codec.py        # ANS wrappers (constriction): encode/decode indices & residuals;
│                   #   adaptive_resid_scale() for fp16/bf16 precision capping
├── format.py       # File I/O: GaussWriter, GaussReader (GAUSS001–004)
└── compress.py     # CLI entry point, multiprocessing workers
```

---

## How It Works

For each weight tensor `w`:

1. **Route** — tensors with `ndim == 1` or `numel < 32 768` go to the raw section
   (lossless). All others enter the GMM pipeline.
2. **GMM Fitting** — fit a K-component mixture of Gaussians using hard EM (pure NumPy).
   Large tensors are subsampled to 200k elements for speed.
3. **Assignment** — assign every weight to its most likely cluster.
4. **Residual Quantization** — compute `r = round((w − μ_cluster) × S)`, clipped to
   INT16 range. Reconstruction error is bounded by `±1/(2×S)`.
   For reduced-precision dtypes, **adaptive-S** caps the scale at the dtype's
   precision floor so sub-epsilon noise is never encoded into the bitstream:

   | dtype | S (effective) | Max error |
   |---|---|---|
   | float32 | 1000 | ±5×10⁻⁴ |
   | float16 | 1000 | ±5×10⁻⁴ |
   | bfloat16 | **128** | ±3.9×10⁻³ |
5. **ANS Encoding** — encode the assignment sequence with a Categorical model and the
   residuals with per-element QuantizedGaussian models, approaching the Shannon entropy
   bound.

Decompression reverses this exactly: decode assignments → decode residuals →
`ŵ = μ_cluster + r / 1000`, then cast back to the original dtype.

---

## Benchmark

Tested on a 24-layer SFT checkpoint (768 hidden dim, 6144 FFN, GQA 4 KV heads,
vocab 32k):

| Method | Type | Ratio | Lossless? |
|---|---|---|---|
| gzip (level 6) | byte-level | ~1.02× | ✅ |
| zstd (level 3) | byte-level | ~1.03× | ✅ |
| INT8 quantization | precision reduction | 4.00× | ❌ |
| **Gauss (K=16)** | **distribution** | **4.90×** | ❌ (±5×10⁻⁴) |
| INT4 quantization | precision reduction | 8.00× | ❌ |

- Compression: **355 s** / Decompression: **95 s** (measured on Google
  Colab CPU — Xeon server-class; consumer CPUs may be faster or slower
  depending on core count and single-core performance; use `--workers`
  to tune parallelism for your host)
- Max per-element reconstruction error: **5.00×10⁻⁴** (tighter than INT8)

---

## Limitations

- **Lossy for large tensors** — bounded error ±5×10⁻⁴; not suitable for bit-exact
  reproduction of large weight matrices.
- **No inter-layer compression** — each tensor is processed independently.
- **ANS stack ordering** — encoding must run in reverse order, imposing a single-pass
  sequential constraint within each tensor's ANS step.
- **Sequential decompression** — parallel decompression across tensors is supported;
  however, within each shard the ANS decode is single-threaded.

---

## Citation

```bibtex
@techreport{seok2026gauss,
  title       = {GAUSS: Distribution-Aware Compression for Neural Network Weights},
  author      = {Seok, Minju and {Claude Sonnet 4.6}},
  year        = {2026},
  month       = {4},
  institution = {Skystarry-AI},
  doi         = {10.5281/zenodo.19676854},
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
