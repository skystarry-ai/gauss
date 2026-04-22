"""
.gauss file format — GMM+ANS compressed weight storage and retrieval.

Two on-disk versions are supported:

GAUSS001 (legacy, single-shard per layer)
------------------------------------------
magic[8]         b"GAUSS001"
n_layers[4]      uint32 LE
--- index section (per-layer metadata) ---
name_len[2], name[n], ndim[1], shape[ndim*4],
K[1], pi[K*8], mu[K*8], sigma[K*8],
idx_words[4], resid_words[4]
--- data section (layers in index order) ---
idx_data[idx_words*4], resid_data[resid_words*4]

GAUSS002 (current, supports multi-shard layers)
------------------------------------------------
magic[8]         b"GAUSS002"
n_layers[4]      uint32 LE
--- index section ---
name_len[2], name[n], ndim[1], shape[ndim*4],
n_shards[2], shard_rows[4],
  repeated n_shards times:
    K[1], pi[K*8], mu[K*8], sigma[K*8], idx_words[4], resid_words[4]
--- data section ---
  repeated n_layers × n_shards:
    idx_data[idx_words*4], resid_data[resid_words*4]

Large 2-D+ layers (> SHARD_THRESHOLD elements) are split row-wise into
shards of SHARD_ROWS rows each; each shard is compressed independently so
all CPU cores stay busy simultaneously.
"""

import struct
from pathlib import Path

import numpy as np

MAGIC = b"GAUSS001"
MAGIC_V2 = b"GAUSS002"
VERSION = 2


class GaussWriter:
    """Write compressed GMM+ANS layers to a .gauss file.

    Example
    -------
    >>> w = GaussWriter("model.gauss")
    >>> w.add(key, shape, K, pi, mu, sigma, idx_compressed, resid_compressed)
    >>> w.add_shards(key, shape, shard_rows, shards)
    >>> w.save()
    >>> print(w.summary())
    """

    def __init__(self, path: str):
        self.path = Path(path)
        # each entry: (key, shape, shard_rows, shards)
        # shards: list of dicts {K, pi, mu, sigma, ic, rc}
        self._layers = []

    def add(
        self,
        key: str,
        shape: tuple,
        K: int,
        pi: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        ic: np.ndarray,
        rc: np.ndarray,
    ):
        """Append a single non-sharded compressed layer.

        Parameters
        ----------
        key:   tensor name used as the lookup key
        shape: original tensor shape
        K:     number of GMM components
        pi:    component weights  (K,) float64
        mu:    component means    (K,) float64
        sigma: component std devs (K,) float64
        ic:    ANS-encoded component indices, uint32 array from constriction
        rc:    ANS-encoded quantized residuals, uint32 array from constriction
        """
        self.add_shards(key, shape, 0, [{
            "K": K,
            "pi": np.asarray(pi, np.float64),
            "mu": np.asarray(mu, np.float64),
            "sigma": np.asarray(sigma, np.float64),
            "ic": np.asarray(ic, np.uint32),
            "rc": np.asarray(rc, np.uint32),
        }])

    def add_shards(
        self,
        key: str,
        shape: tuple,
        shard_rows: int,
        shards: list,
    ):
        """Append a (possibly multi-shard) compressed layer.

        Parameters
        ----------
        key:        tensor name used as the lookup key
        shape:      original tensor shape
        shard_rows: rows per shard; 0 for non-sharded layers
        shards:     list of dicts, each with keys K, pi, mu, sigma, ic, rc
        """
        normalized = []
        for s in shards:
            normalized.append({
                "K": int(s["K"]),
                "pi": np.asarray(s["pi"], np.float64),
                "mu": np.asarray(s["mu"], np.float64),
                "sigma": np.asarray(s["sigma"], np.float64),
                "ic": np.asarray(s["ic"], np.uint32),
                "rc": np.asarray(s["rc"], np.uint32),
            })
        self._layers.append((key, shape, int(shard_rows), normalized))

    def save(self) -> int:
        """Serialize all layers to disk and return the file size in bytes."""
        with open(self.path, "wb") as f:
            # header
            f.write(MAGIC_V2)
            f.write(struct.pack("<I", len(self._layers)))

            # index section: per-layer metadata
            for key, shape, shard_rows, shards in self._layers:
                nb = key.encode("utf-8")
                f.write(struct.pack("<H", len(nb)))
                f.write(nb)
                f.write(struct.pack("<B", len(shape)))
                for s in shape:
                    f.write(struct.pack("<I", s))
                f.write(struct.pack("<HI", len(shards), shard_rows))
                for sh in shards:
                    K = sh["K"]
                    f.write(struct.pack("<B", K))
                    sh["pi"].tofile(f)
                    sh["mu"].tofile(f)
                    sh["sigma"].tofile(f)
                    f.write(struct.pack("<II", len(sh["ic"]), len(sh["rc"])))

            # data section: compressed payloads in index order
            for _, _, _, shards in self._layers:
                for sh in shards:
                    sh["ic"].tofile(f)
                    sh["rc"].tofile(f)

        return self.path.stat().st_size

    def summary(self) -> str:
        """Return a human-readable compression summary string."""
        total_orig = sum(int(np.prod(s)) * 4 for _, s, *_ in self._layers)
        total_comp = self.path.stat().st_size
        n_sharded = sum(1 for _, _, _, shards in self._layers if len(shards) > 1)
        shard_note = f", {n_sharded} sharded" if n_sharded else ""
        return (
            f"{len(self._layers)} layers{shard_note} | "
            f"original {total_orig / 1024 / 1024:.1f} MB → "
            f"compressed {total_comp / 1024 / 1024:.1f} MB | "
            f"ratio {total_orig / total_comp:.3f}x"
        )


class GaussReader:
    """Read and decompress layers from a .gauss file.

    Supports both GAUSS001 (legacy) and GAUSS002 (sharded) formats.

    Example
    -------
    >>> r = GaussReader("model.gauss")
    >>> print(r.keys())
    >>> weights = r.decompress("layers.0.ffn.w1.weight")  # np.ndarray f32
    >>> state_dict = r.decompress_all()                    # {key: np.ndarray}
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._index: list = []
        self._data_offset: int = 0
        self._parse_index()

    def _parse_index(self):
        """Parse the index section from the file header into ``self._index``."""
        with open(self.path, "rb") as f:
            magic = f.read(8)
            if magic == MAGIC:
                version = 1
            elif magic == MAGIC_V2:
                version = 2
            else:
                raise ValueError(f"Invalid .gauss file: magic={magic!r}")
            n = struct.unpack("<I", f.read(4))[0]

            for _ in range(n):
                name_len = struct.unpack("<H", f.read(2))[0]
                key = f.read(name_len).decode("utf-8")
                ndim = struct.unpack("<B", f.read(1))[0]
                shape = tuple(
                    struct.unpack("<I", f.read(4))[0] for _ in range(ndim)
                )

                if version == 1:
                    # legacy single-shard layout
                    K = struct.unpack("<B", f.read(1))[0]
                    pi = np.frombuffer(f.read(K * 8), np.float64).copy()
                    mu = np.frombuffer(f.read(K * 8), np.float64).copy()
                    sigma = np.frombuffer(f.read(K * 8), np.float64).copy()
                    ic_words, rc_words = struct.unpack("<II", f.read(8))
                    shards = [dict(
                        K=K, pi=pi, mu=mu, sigma=sigma,
                        ic_words=ic_words, rc_words=rc_words,
                    )]
                    shard_rows = 0
                else:
                    n_shards, shard_rows = struct.unpack("<HI", f.read(6))
                    shards = []
                    for _ in range(n_shards):
                        K = struct.unpack("<B", f.read(1))[0]
                        pi = np.frombuffer(f.read(K * 8), np.float64).copy()
                        mu = np.frombuffer(f.read(K * 8), np.float64).copy()
                        sigma = np.frombuffer(f.read(K * 8), np.float64).copy()
                        ic_words, rc_words = struct.unpack("<II", f.read(8))
                        shards.append(dict(
                            K=K, pi=pi, mu=mu, sigma=sigma,
                            ic_words=ic_words, rc_words=rc_words,
                        ))

                self._index.append(dict(
                    key=key,
                    shape=shape,
                    shard_rows=shard_rows,
                    shards=shards,
                ))

            self._data_offset = f.tell()

    def keys(self) -> list:
        """Return a list of all layer names stored in this file."""
        return [e["key"] for e in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def decompress(self, key: str) -> np.ndarray:
        """Decompress a single layer and return a float32 numpy array.

        For sharded layers all shards are decoded and concatenated before
        reshaping to the original tensor shape.
        """
        import constriction

        try:
            from .gmm_compress import RESID_BOUND, RESID_SCALE
        except ImportError:
            from gmm_compress import RESID_BOUND, RESID_SCALE  # script mode

        e = next((x for x in self._index if x["key"] == key), None)
        if e is None:
            raise KeyError(key)

        # compute byte offset for this layer's data block
        offset = self._data_offset
        for x in self._index:
            if x["key"] == key:
                break
            for sh in x["shards"]:
                offset += (sh["ic_words"] + sh["rc_words"]) * 4

        shape = e["shape"]
        shard_rows = e["shard_rows"]
        shards = e["shards"]
        n_shards = len(shards)
        total_elements = int(np.prod(shape))

        # compute per-shard element count
        if n_shards == 1:
            shard_n_elements = [total_elements]
        else:
            row_size = int(np.prod(shape[1:]))
            shard_n_elements = [shard_rows * row_size] * (n_shards - 1)
            shard_n_elements.append(
                total_elements - (n_shards - 1) * shard_rows * row_size
            )

        all_weights = []
        with open(self.path, "rb") as f:
            f.seek(offset)
            for sh, N in zip(shards, shard_n_elements):
                ic = np.frombuffer(f.read(sh["ic_words"] * 4), np.uint32).copy()
                rc = np.frombuffer(f.read(sh["rc_words"] * 4), np.uint32).copy()

                pi, mu, sigma = sh["pi"], sh["mu"], sh["sigma"]

                probs = (pi / pi.sum()).astype(np.float32)
                model = constriction.stream.model.Categorical(
                    probs, perfect=False
                )
                ans = constriction.stream.stack.AnsCoder(ic)
                asgn = ans.decode(model, N)

                sigma_a = sigma[asgn]
                means = np.zeros(N, np.float64)
                stds = np.maximum(sigma_a * RESID_SCALE, 0.5)
                model_g = constriction.stream.model.QuantizedGaussian(
                    -RESID_BOUND, RESID_BOUND
                )
                ans2 = constriction.stream.stack.AnsCoder(rc)
                resid_q = ans2.decode(model_g, means, stds)

                all_weights.append(
                    (mu[asgn] + resid_q / RESID_SCALE).astype(np.float32)
                )

        return np.concatenate(all_weights).reshape(shape)

    def decompress_all(self) -> dict:
        """Decompress all layers and return a ``{key: np.ndarray}`` dict."""
        return {e["key"]: self.decompress(e["key"]) for e in self._index}


# ---------------------------------------------------------------------------
# CLI: compress / decompress / info
# ---------------------------------------------------------------------------

def _worker_save(task):
    """Compress a single layer or row-shard; return a result dict.

    task = (key, shard_idx, n_shards, data_np, K, max_iter, max_fit)
    """
    import time
    import traceback

    import numpy as np

    try:
        from .gmm_compress import (
            RESID_BOUND,
            RESID_SCALE,
            assign_all,
            encode_indices,
            encode_residuals,
            fit_gmm_fast,
        )
    except ImportError:
        from gmm_compress import (
            RESID_BOUND,
            RESID_SCALE,
            assign_all,
            encode_indices,
            encode_residuals,
            fit_gmm_fast,
        )

    key, shard_idx, n_shards, data_np, K, max_iter, max_fit = task
    try:
        data = data_np.astype(np.float64)
        N = len(data)
        t0 = time.perf_counter()
        if N > max_fit:
            fit_data = data[np.random.choice(N, max_fit, replace=False)]
        else:
            fit_data = data
        pi, mu, sigma, n_iter = fit_gmm_fast(fit_data, K, max_iter)
        asgn = assign_all(data, pi, mu, sigma)
        mu_a = mu[asgn]
        sigma_a = sigma[asgn]
        resid_q = np.clip(
            np.round((data - mu_a) * RESID_SCALE).astype(np.int32),
            -RESID_BOUND,
            RESID_BOUND,
        )
        ic = encode_indices(asgn, pi)
        rc = encode_residuals(resid_q, sigma_a)
        orig = N * 4
        comp = (len(ic) + len(rc)) * 4 + K * 3 * 8
        return {
            "key": key, "shard_idx": shard_idx, "n_shards": n_shards,
            "pi": pi, "mu": mu, "sigma": sigma,
            "ic": ic, "rc": rc, "K": K,
            "orig": orig, "comp": comp,
            "ratio": orig / comp, "n_iter": n_iter,
            "t": time.perf_counter() - t0, "error": None,
        }
    except Exception:
        return {
            "key": key, "shard_idx": shard_idx, "n_shards": n_shards,
            "error": traceback.format_exc(),
        }


def main():
    import argparse
    import math
    import time
    from multiprocessing import Pool, cpu_count

    import torch
    from safetensors.torch import load_file, save_file

    try:
        from .gmm_compress import SHARD_ROWS, SHARD_THRESHOLD
    except ImportError:
        from gmm_compress import SHARD_ROWS, SHARD_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Gauss — GMM+ANS weight compression CLI"
    )
    sub = parser.add_subparsers(dest="cmd")

    # compress subcommand
    p_c = sub.add_parser("compress", help=".safetensors → .gauss")
    p_c.add_argument("input", help="input .safetensors file")
    p_c.add_argument("output", help="output .gauss file")
    p_c.add_argument("--K", type=int, default=16)
    p_c.add_argument("--max-iter", type=int, default=100)
    p_c.add_argument("--workers", type=int, default=None)
    p_c.add_argument(
        "--shard-threshold", type=int, default=SHARD_THRESHOLD,
        help=(
            "shard 2-D+ layers with more elements than this "
            f"(default: {SHARD_THRESHOLD:,})"
        ),
    )
    p_c.add_argument(
        "--shard-rows", type=int, default=SHARD_ROWS,
        help=f"rows per shard for large layers (default: {SHARD_ROWS})",
    )

    # decompress subcommand
    p_d = sub.add_parser("decompress", help=".gauss → .safetensors")
    p_d.add_argument("input", help="input .gauss file")
    p_d.add_argument("output", help="output .safetensors file")

    # info subcommand
    p_i = sub.add_parser("info", help="display .gauss file metadata")
    p_i.add_argument("file")

    args = parser.parse_args()

    if args.cmd == "info":
        r = GaussReader(args.file)
        size = Path(args.file).stat().st_size
        print(f"File:   {args.file}")
        print(f"Size:   {size / 1024 / 1024:.2f} MB")
        print(f"Layers: {len(r)}")
        for e in r._index:
            n = int(np.prod(e["shape"]))
            comp = sum(
                s["ic_words"] + s["rc_words"] for s in e["shards"]
            ) * 4
            n_shards = len(e["shards"])
            shard_tag = f" [×{n_shards} shards]" if n_shards > 1 else ""
            K = e["shards"][0]["K"]
            print(
                f"  {e['key']:<50}  {n * 4 // 1024:>6}KB → "
                f"{comp // 1024:>5}KB  {n * 4 / comp:.3f}x"
                f"  K={K}{shard_tag}"
            )

    elif args.cmd == "compress":
        print(f"Loading {args.input} ...")
        sd = load_file(args.input)

        # sort largest-first for better initial load balancing
        keys = sorted(sd.keys(), key=lambda k: sd[k].numel(), reverse=True)

        n_workers = args.workers or cpu_count()
        shard_threshold = args.shard_threshold
        shard_rows = args.shard_rows

        # build task list; large 2-D+ layers are split into row-wise shards
        tasks = []
        shapes = {k: tuple(sd[k].shape) for k in keys}
        layer_shard_info = {}  # key -> (n_shards, actual_shard_rows)

        for k in keys:
            t = sd[k]
            flat = t.float().numpy().flatten()
            numel = t.numel()

            if numel > shard_threshold and t.ndim >= 2:
                n_rows = t.shape[0]
                # cap shard_rows so we always have at least one row per shard
                actual_sr = min(shard_rows, n_rows)
                n_shards = math.ceil(n_rows / actual_sr)
                row_size = numel // n_rows
                layer_shard_info[k] = (n_shards, actual_sr)
                for i in range(n_shards):
                    start = i * actual_sr * row_size
                    end = min(start + actual_sr * row_size, numel)
                    tasks.append((
                        k, i, n_shards,
                        flat[start:end],
                        args.K, args.max_iter, 200_000,
                    ))
            else:
                layer_shard_info[k] = (1, 0)
                tasks.append((k, 0, 1, flat, args.K, args.max_iter, 200_000))

        n_sharded_layers = sum(
            1 for v in layer_shard_info.values() if v[0] > 1
        )
        print(
            f"Layers: {len(keys)} ({n_sharded_layers} sharded) | "
            f"tasks: {len(tasks)} | K={args.K} | workers={n_workers}"
        )

        # pending[key] collects per-shard results; entry removed once complete
        pending = {
            k: [None] * info[0] for k, info in layer_shard_info.items()
        }
        writer = GaussWriter(args.output)
        total_orig = total_comp = 0
        completed = []
        t0 = time.perf_counter()

        with Pool(n_workers) as pool:
            for res in pool.imap_unordered(_worker_save, tasks):
                if res.get("error"):
                    print(
                        f"  ERROR [{res['key']} shard {res.get('shard_idx', 0)}]"
                        f"\n{res['error'][:200]}"
                    )
                    continue

                key = res["key"]
                pending[key][res["shard_idx"]] = res

                # flush to writer only when every shard of the layer is ready
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
                    layer_shard_info[key][1],
                    [
                        {
                            "K": s["K"], "pi": s["pi"], "mu": s["mu"],
                            "sigma": s["sigma"], "ic": s["ic"], "rc": s["rc"],
                        }
                        for s in shard_results
                    ],
                )
                completed.append(key)

                shard_tag = f" [{n_sh} shards]" if n_sh > 1 else ""
                # report slowest shard wall-time (shards run in parallel)
                elapsed_str = f"{max(s['t'] for s in shard_results):.1f}s"
                print(
                    f"  [{len(completed):4d}/{len(keys)}] {key:<46}"
                    f"  {orig_sum / comp_sum:.3f}x"
                    f"  iter={shard_results[0]['n_iter']:3d}"
                    f"{shard_tag}  {elapsed_str}"
                )

        saved = writer.save()
        elapsed = time.perf_counter() - t0
        print(f"\nSaved: {args.output}  ({saved / 1024 / 1024:.1f} MB)")
        print(
            f"Overall ratio: {total_orig / total_comp:.3f}x  "
            f"({total_orig // 1024 // 1024}MB → {saved // 1024 // 1024}MB)  "
            f"total {elapsed:.0f}s"
        )

    elif args.cmd == "decompress":
        print(f"Loading {args.input} ...")
        r = GaussReader(args.input)
        t0 = time.perf_counter()
        sd = {}
        for i, key in enumerate(r.keys(), 1):
            arr = r.decompress(key)
            sd[key] = torch.from_numpy(arr)
            print(f"  [{i:4d}/{len(r)}] {key}")
        save_file(sd, args.output)
        size = Path(args.output).stat().st_size
        print(f"\nSaved: {args.output}  ({size / 1024 / 1024:.1f} MB)")
        print(f"Decompression complete  ({time.perf_counter() - t0:.1f}s)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
