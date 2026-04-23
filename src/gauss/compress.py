"""
CLI entry point and multiprocessing workers for Gauss compression.

Design notes
------------
- Layers with numel < RAW_THRESHOLD or ndim == 1 are stored losslessly in
  the raw section rather than being skipped entirely.
- Original tensor dtype is preserved; the file records a dtype_id so
  decompression can restore the exact original precision.
- Large 2-D+ layers (> SHARD_THRESHOLD elements) are split row-wise into
  independent shards for better multi-core load balancing.
- All workers import their dependencies locally so the pool spawns cleanly.
"""

import math
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from .codec import (
    RESID_BOUND,
    RESID_SCALE,
    encode_indices,
    encode_residuals,
)
from .gmm import assign_all
from .format import DTYPE_MAP, GaussReader, GaussWriter, _DTYPE_NUMPY

__all__ =[
    "SHARD_THRESHOLD",
    "SHARD_ROWS",
    "RAW_THRESHOLD",
    "compress_file",
    "decompress_file",
]

# Shard 2-D+ layers larger than this many elements.
SHARD_THRESHOLD: int = 5_000_000

# Number of rows per shard when sharding is applied.
SHARD_ROWS: int = 4096

# Layers with fewer elements than this are stored losslessly (raw section).
# Also, all 1-D layers are always stored raw regardless of size.
RAW_THRESHOLD: int = 32768


# ---------------------------------------------------------------------------
# Multiprocessing worker — GMM compression
# ---------------------------------------------------------------------------

def _worker_compress(task: tuple) -> dict:
    """Compress one layer shard; designed to run inside a Pool worker.

    Parameters
    ----------
    task : tuple of
        (key, shard_idx, n_shards, data_np, K, max_iter, max_fit_samples, dtype_str)

    Returns
    -------
    dict with keys: key, shard_idx, n_shards, pi, mu, sigma, ic, rc, K,
                    orig, comp, ratio, n_iter, t, error
    """
    import time
    import traceback

    import numpy as np

    from .codec import (
        RESID_BOUND,
        RESID_SCALE,
        adaptive_resid_scale,
        encode_indices,
        encode_residuals,
    )
    from .gmm import fit_gmm_fast, assign_all

    key, shard_idx, n_shards, data_np, K, max_iter, max_fit, dtype_str = task
    try:
        data = data_np.astype(np.float64)
        N = len(data)
        t0 = time.perf_counter()

        # Subsample for GMM fitting to keep memory and runtime bounded.
        if N > max_fit:
            idx = np.random.choice(N, max_fit, replace=False)
            fit_data = data[idx]
        else:
            fit_data = data

        # Clamp quantization scale to the precision floor of the source dtype.
        # Encoding residuals finer than the dtype's ULP wastes bitstream space.
        eff_scale = adaptive_resid_scale(dtype_str, RESID_SCALE)

        pi, mu, sigma, n_iter = fit_gmm_fast(fit_data, K, max_iter)
        asgn = assign_all(data, pi, mu, sigma)
        mu_a = mu[asgn]
        sigma_a = sigma[asgn]
        resid_q = np.clip(
            np.round((data - mu_a) * eff_scale).astype(np.int32),
            -RESID_BOUND,
            RESID_BOUND,
        )
        ic = encode_indices(asgn, pi)
        rc = encode_residuals(resid_q, sigma_a, scale=eff_scale)
        orig = N * (2 if dtype_str in ("bfloat16", "float16") else 4)
        comp = (len(ic) + len(rc)) * 4 + K * 3 * 8
        return {
            "key": key, "shard_idx": shard_idx, "n_shards": n_shards,
            "pi": pi, "mu": mu, "sigma": sigma,
            "ic": ic, "rc": rc, "K": K,
            "eff_scale": eff_scale,
            "orig": orig, "comp": comp,
            "ratio": orig / comp, "n_iter": n_iter,
            "t": time.perf_counter() - t0,
            "error": None,
        }
    except Exception:
        return {
            "key": key, "shard_idx": shard_idx, "n_shards": n_shards,
            "error": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Multiprocessing worker — decompression
# ---------------------------------------------------------------------------

def _worker_decompress(args: tuple) -> tuple:
    """Decompress one tensor from a .gauss file.

    Designed to run inside a ``multiprocessing.Pool``.

    Parameters
    ----------
    args : tuple of (path_str, data_offset, raw_data_offset, entry_dict,
                     is_raw)
        path_str         – absolute path to the .gauss file.
        data_offset      – byte offset of the compressed data section.
        raw_data_offset  – byte offset of the raw data section.
        entry_dict       – one element of GaussReader._index or _raw_index.
        is_raw           – True if this is a raw (lossless) entry.

    Returns
    -------
    (key, np.ndarray) with dtype matching the original tensor.
    """
    import numpy as np

    path, data_offset, raw_data_offset, entry, is_raw = args
    key = entry["key"]
    shape = entry["shape"]
    dtype_id = entry["dtype_id"]

    if is_raw:
        # Compute byte offset in the raw data section.
        np_dtype = _DTYPE_NUMPY[dtype_id]
        if dtype_id == 2:  # bfloat16
            np_dtype = np.int16

        with open(path, "rb") as f:
            f.seek(raw_data_offset)
            raw = f.read(entry["raw_n_bytes"])
        arr = np.frombuffer(raw, dtype=np_dtype).copy().reshape(shape)
        return key, arr

    # GMM-compressed path.
    import constriction

    from .codec import RESID_BOUND, RESID_SCALE

    shard_rows = entry["shard_rows"]
    shards = entry["shards"]
    n_shards = len(shards)
    total_elements = int(np.prod(shape))

    if n_shards == 1:
        shard_n = [total_elements]
    else:
        row_size = int(np.prod(shape[1:]))
        shard_n =[shard_rows * row_size] * (n_shards - 1)
        shard_n.append(
            total_elements - (n_shards - 1) * shard_rows * row_size
        )

    out_chunks =[]
    with open(path, "rb") as f:
        f.seek(data_offset)
        for sh, N in zip(shards, shard_n):
            ic = np.frombuffer(
                f.read(sh["ic_words"] * 4), np.uint32
            ).copy()
            rc = np.frombuffer(
                f.read(sh["rc_words"] * 4), np.uint32
            ).copy()

            pi, mu, sigma = sh["pi"], sh["mu"], sh["sigma"]
            # Use the scale stored in the shard (v4) or default (legacy).
            s = sh.get("resid_scale", RESID_SCALE)
            probs = (pi / pi.sum()).astype(np.float32)
            model_cat = constriction.stream.model.Categorical(
                probs, perfect=False
            )
            asgn = constriction.stream.stack.AnsCoder(ic).decode(
                model_cat, N
            )
            sigma_a = sigma[asgn]
            means = np.zeros(N, np.float64)
            stds = np.maximum(sigma_a * s, 0.5)
            model_g = constriction.stream.model.QuantizedGaussian(
                -RESID_BOUND, RESID_BOUND
            )
            resid_q = constriction.stream.stack.AnsCoder(rc).decode(
                model_g, means, stds
            )
            out_chunks.append(mu[asgn] + resid_q / s)

    np_dtype = _DTYPE_NUMPY[dtype_id]
    flat = np.concatenate(out_chunks).astype(np_dtype)
    return key, flat.reshape(shape)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def compress_file(
    input_path: str,
    output_path: str,
    K: int = 16,
    max_iter: int = 100,
    max_fit_samples: int = 200_000,
    n_workers: int = None,
    shard_threshold: int = SHARD_THRESHOLD,
    shard_rows: int = SHARD_ROWS,
    raw_threshold: int = RAW_THRESHOLD,
    layer_filter: list = None,
) -> dict:
    """Compress a .safetensors file to .gauss format.

    Parameters
    ----------
    input_path:       Path to the input .safetensors file.
    output_path:      Path for the output .gauss file.
    K:                Number of GMM components per tensor.
    max_iter:         Maximum hard-EM iterations.
    max_fit_samples:  Maximum elements used for GMM fitting (subsampled).
    n_workers:        Worker processes; defaults to all CPU cores.
    shard_threshold:  Element count above which 2-D+ layers are sharded.
    shard_rows:       Rows per shard.
    raw_threshold:    Layers with fewer elements are stored raw (lossless).
                      1-D layers are always stored raw regardless.
    layer_filter:     If given, compress only these layer names.

    Returns
    -------
    dict with keys: saved_bytes, total_orig, total_comp, elapsed,
                    n_compressed, n_raw, n_workers.
    """
    import torch
    from safetensors.torch import load_file

    n_workers = n_workers or cpu_count()
    print(f"Loading {input_path} ...")
    sd = load_file(input_path)

    if layer_filter is not None:
        keys = [k for k in layer_filter if k in sd]
    else:
        keys = list(sd.keys())

    # Sort largest-first for better initial load balancing.
    keys.sort(key=lambda k: sd[k].numel(), reverse=True)

    # Separate raw (lossless) layers from compressed layers.
    compress_keys =[]
    raw_keys = []
    for k in keys:
        t = sd[k]
        if t.ndim == 1 or t.numel() < raw_threshold:
            raw_keys.append(k)
        else:
            compress_keys.append(k)

    # Build shard tasks for compressed layers.
    tasks = []
    shapes = {k: tuple(sd[k].shape) for k in compress_keys}
    dtype_ids = {
        k: DTYPE_MAP.get(str(sd[k].dtype), 0) for k in compress_keys
    }
    # Plain dtype name strings passed to adaptive_resid_scale (e.g. "float16").
    dtype_strs = {k: str(sd[k].dtype).split(".")[-1] for k in compress_keys}
    layer_shard_info = {}  # key -> (n_shards, actual_shard_rows)

    for k in compress_keys:
        t = sd[k]
        flat = t.float().numpy().flatten()
        numel = t.numel()
        dstr = dtype_strs[k]

        if numel > shard_threshold and t.ndim >= 2:
            n_rows = t.shape[0]
            actual_sr = min(shard_rows, n_rows)
            n_sh = math.ceil(n_rows / actual_sr)
            row_size = numel // n_rows
            layer_shard_info[k] = (n_sh, actual_sr)
            for i in range(n_sh):
                start = i * actual_sr * row_size
                end = min(start + actual_sr * row_size, numel)
                tasks.append((
                    k, i, n_sh, flat[start:end],
                    K, max_iter, max_fit_samples, dstr,
                ))
        else:
            layer_shard_info[k] = (1, 0)
            tasks.append((k, 0, 1, flat, K, max_iter, max_fit_samples, dstr))

    n_sharded = sum(1 for v in layer_shard_info.values() if v[0] > 1)
    print(
        f"Layers: {len(compress_keys)} compressed ({n_sharded} sharded), "
        f"{len(raw_keys)} raw | "
        f"tasks: {len(tasks)} | K={K} | workers={n_workers}"
    )

    pending = {
        k: [None] * layer_shard_info[k][0] for k in compress_keys
    }
    writer = GaussWriter(output_path)
    total_orig = total_comp = 0
    completed =[]
    t0 = time.perf_counter()

    with Pool(n_workers) as pool:
        for res in pool.imap_unordered(_worker_compress, tasks):
            if res.get("error"):
                print(
                    f"  ERROR [{res['key']} shard {res.get('shard_idx', 0)}]"
                    f"\n{res['error'][:300]}"
                )
                continue

            key = res["key"]
            pending[key][res["shard_idx"]] = res

            # Flush to writer only when every shard of the layer is ready.
            if any(s is None for s in pending[key]):
                continue

            shard_results = pending.pop(key)
            n_sh = len(shard_results)
            orig_sum = sum(s["orig"] for s in shard_results)
            comp_sum = sum(s["comp"] for s in shard_results)
            total_orig += orig_sum
            total_comp += comp_sum

            writer.add_shards(
                key,
                shapes[key],
                dtype_ids[key],
                layer_shard_info[key][1],[
                    {
                        "K": s["K"], "pi": s["pi"], "mu": s["mu"],
                        "sigma": s["sigma"], "ic": s["ic"], "rc": s["rc"],
                        # Propagate the effective scale so GaussWriter can
                        # record it in the GAUSS004 index section.
                        "resid_scale": s["eff_scale"],
                    }
                    for s in shard_results
                ],
            )
            completed.append(key)

            shard_tag = f"[{n_sh} shards]" if n_sh > 1 else ""
            elapsed_str = f"{max(s['t'] for s in shard_results):.1f}s"
            print(
                f"  [{len(completed):4d}/{len(compress_keys)}] {key:<46}"
                f"  {orig_sum / comp_sum:.3f}x"
                f"  iter={shard_results[0]['n_iter']:3d}"
                f"{shard_tag}  {elapsed_str}"
            )

    # Add raw layers.
    for k in raw_keys:
        t = sd[k]
        dtype_id = DTYPE_MAP.get(str(t.dtype), 0)
        # Store exact raw bytes for bfloat16 via int16 view to prevent loss
        if t.dtype == torch.bfloat16:
            arr = t.view(torch.int16).numpy()
        else:
            arr = t.numpy()
        raw_bytes = arr.tobytes()
        total_orig += len(raw_bytes)
        total_comp += len(raw_bytes)
        writer.add_raw(k, tuple(t.shape), dtype_id, raw_bytes)
        print(f"  [raw] {k}")

    saved = writer.save()
    elapsed = time.perf_counter() - t0
    print(f"\nSaved: {output_path}  ({saved / 1024 / 1024:.1f} MB)")
    if total_comp > 0:
        print(
            f"Overall ratio: {total_orig / total_comp:.3f}x  "
            f"({total_orig // 1024 // 1024}MB → "
            f"{saved // 1024 // 1024}MB)  "
            f"total {elapsed:.0f}s"
        )
    return {
        "saved_bytes": saved,
        "total_orig": total_orig,
        "total_comp": total_comp,
        "elapsed": elapsed,
        "n_compressed": len(compress_keys),
        "n_raw": len(raw_keys),
        "n_workers": n_workers,
    }


def decompress_file(
    input_path: str,
    output_path: str,
    n_workers: int = None,
) -> dict:
    """Decompress a .gauss file back to .safetensors format.

    Each tensor is restored to its original dtype as recorded in the file.

    Parameters
    ----------
    input_path:  Path to the input .gauss file.
    output_path: Path for the output .safetensors file.
    n_workers:   Worker processes; defaults to all CPU cores.

    Returns
    -------
    dict with keys: saved_bytes, elapsed, n_layers, n_workers.
    """
    import torch
    from safetensors.torch import save_file

    from .format import DTYPE_ID_MAP

    n_workers = n_workers or cpu_count()
    print(f"Loading {input_path} ...")
    reader = GaussReader(input_path)

    # Pre-compute byte offsets so workers can seek directly.
    tasks =[]
    offset = reader._data_offset
    for entry in reader._index:
        tasks.append((
            str(reader.path), offset, 0, entry, False,
        ))
        for sh in entry["shards"]:
            offset += (sh["ic_words"] + sh["rc_words"]) * 4

    # Raw tasks: each gets its own byte offset pre-computed.
    raw_offset = reader._raw_data_offset
    for entry in reader._raw_index:
        tasks.append((
            str(reader.path), 0, raw_offset, entry, True,
        ))
        raw_offset += entry["raw_n_bytes"]

    n_layers = len(reader)
    input_size = Path(input_path).stat().st_size
    print(
        f"Layers: {n_layers}  |  workers={n_workers}  |  "
        f"input {input_size / 1024 / 1024:.1f} MB"
    )

    t0 = time.perf_counter()
    sd = {}
    with Pool(n_workers) as pool:
        for i, (key, arr) in enumerate(
            pool.imap_unordered(_worker_decompress, tasks), 1
        ):
            # Convert numpy array back to the correct torch dtype.
            dtype_id = next(
                (e["dtype_id"] for e in reader._index + reader._raw_index
                 if e["key"] == key),
                0,
            )
            torch_dtype_str = DTYPE_ID_MAP.get(dtype_id, "torch.float32")
            torch_dtype = getattr(torch, torch_dtype_str.replace("torch.", ""))

            # View correctly preserved bytes as bfloat16
            if torch_dtype == torch.bfloat16 and arr.dtype == np.int16:
                sd[key] = torch.from_numpy(arr).view(torch.bfloat16)
            else:
                sd[key] = torch.from_numpy(arr).to(torch_dtype)
                
            print(f"  [{i:4d}/{n_layers}] {key}")

    save_file(sd, output_path)
    saved = Path(output_path).stat().st_size
    elapsed = time.perf_counter() - t0
    print(f"\nSaved: {output_path}  ({saved / 1024 / 1024:.1f} MB)")
    print(f"Decompression complete  ({elapsed:.1f}s)  workers={n_workers}")
    return {
        "saved_bytes": saved,
        "elapsed": elapsed,
        "n_layers": n_layers,
        "n_workers": n_workers,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``gauss`` command-line tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Gauss v0.1.0 — GMM+ANS LLM weight compression"
    )
    sub = parser.add_subparsers(dest="cmd")

    # compress subcommand
    p_c = sub.add_parser("compress", help=".safetensors → .gauss")
    p_c.add_argument("input", help="input .safetensors file")
    p_c.add_argument("output", help="output .gauss file")
    p_c.add_argument("--K", type=int, default=16,
                     help="GMM components per tensor (default: 16)")
    p_c.add_argument("--max-iter", type=int, default=100,
                     help="max hard-EM iterations (default: 100)")
    p_c.add_argument("--workers", type=int, default=None,
                     help="worker processes (default: all cores)")
    p_c.add_argument(
        "--shard-threshold", type=int, default=SHARD_THRESHOLD,
        help=(
            f"shard 2-D+ layers with more elements than this "
            f"(default: {SHARD_THRESHOLD:,})"
        ),
    )
    p_c.add_argument(
        "--shard-rows", type=int, default=SHARD_ROWS,
        help=f"rows per shard (default: {SHARD_ROWS})",
    )
    p_c.add_argument(
        "--raw-threshold", type=int, default=RAW_THRESHOLD,
        help=(
            f"store layers with fewer elements losslessly "
            f"(default: {RAW_THRESHOLD:,}); "
            "1-D layers are always stored raw"
        ),
    )
    p_c.add_argument(
        "--layers", type=str, default=None,
        help="comma-separated layer names (all eligible layers if omitted)",
    )

    # decompress subcommand
    p_d = sub.add_parser("decompress", help=".gauss → .safetensors")
    p_d.add_argument("input", help="input .gauss file")
    p_d.add_argument("output", help="output .safetensors file")
    p_d.add_argument("--workers", type=int, default=None,
                     help="worker processes (default: all cores)")

    # info subcommand
    p_i = sub.add_parser("info", help="display .gauss metadata")
    p_i.add_argument("file")

    args = parser.parse_args()

    if args.cmd == "compress":
        layer_filter = None
        if args.layers:
            layer_filter =[k.strip() for k in args.layers.split(",")]
        compress_file(
            args.input, args.output,
            K=args.K,
            max_iter=args.max_iter,
            n_workers=args.workers,
            shard_threshold=args.shard_threshold,
            shard_rows=args.shard_rows,
            raw_threshold=args.raw_threshold,
            layer_filter=layer_filter,
        )

    elif args.cmd == "decompress":
        decompress_file(args.input, args.output, n_workers=args.workers)

    elif args.cmd == "info":
        _cmd_info(args.file)

    else:
        parser.print_help()


def _cmd_info(file_path: str) -> None:
    """Print per-layer metadata without decompressing anything."""
    reader = GaussReader(file_path)
    size = Path(file_path).stat().st_size
    from .codec import RESID_SCALE
    from .format import DTYPE_ID_MAP

    print(f"File:    {file_path}")
    print(f"Size:    {size / 1024 / 1024:.2f} MB")
    print(f"Layers:  {len(reader)}  "
          f"({len(reader._index)} compressed, "
          f"{len(reader._raw_index)} raw)")
    print()

    for e in reader._index:
        n = int(np.prod(e["shape"]))
        comp = sum(
            s["ic_words"] + s["rc_words"] for s in e["shards"]
        ) * 4
        n_shards = len(e["shards"])
        shard_tag = f"[×{n_shards} shards]" if n_shards > 1 else ""
        K = e["shards"][0]["K"]
        # resid_scale is uniform per layer in practice; show the first shard's.
        s_eff = e["shards"][0].get("resid_scale", RESID_SCALE)
        dtype_str = DTYPE_ID_MAP.get(e["dtype_id"], "?")
        bpe = 2 if dtype_str in ("bfloat16", "float16") else 4
        print(
            f"  {e['key']:<50}  {n * bpe // 1024:>6}KB → "
            f"{comp // 1024:>5}KB  {n * bpe / comp:.3f}x"
            f"  K={K}  S={s_eff}  {dtype_str}{shard_tag}"
        )

    if reader._raw_index:
        print()
        for e in reader._raw_index:
            dtype_str = DTYPE_ID_MAP.get(e["dtype_id"], "?")
            n = int(np.prod(e["shape"]))
            print(
                f"  {e['key']:<50}  {e['raw_n_bytes'] // 1024:>6}KB  "
                f"[raw / lossless]  {dtype_str}"
            )


if __name__ == "__main__":
    main()
