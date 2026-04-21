"""
.gauss file format — GMM+ANS compressed weight storage and retrieval.

Binary layout
-------------
magic[8]         b"GAUSS001"
n_layers[4]      uint32 LE
--- index section (per-layer metadata) ---
name_len[2], name[n], ndim[1], shape[ndim*4],
K[1], pi[K*8], mu[K*8], sigma[K*8],
idx_words[4], resid_words[4]
--- data section (layers in index order) ---
idx_data[idx_words*4], resid_data[resid_words*4]
"""

import struct
from pathlib import Path

import numpy as np

MAGIC = b"GAUSS001"
VERSION = 1


class GaussWriter:
    """Write compressed GMM+ANS layers to a .gauss file.

    Example
    -------
    >>> w = GaussWriter("model.gauss")
    >>> w.add(key, shape, K, pi, mu, sigma, idx_compressed, resid_compressed)
    >>> w.save()
    >>> print(w.summary())
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._layers = []  # list of (key, shape, K, pi, mu, sigma, ic, rc)

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
        """Append a single compressed layer.

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
        self._layers.append((
            key,
            shape,
            K,
            np.asarray(pi, np.float64),
            np.asarray(mu, np.float64),
            np.asarray(sigma, np.float64),
            np.asarray(ic, np.uint32),
            np.asarray(rc, np.uint32),
        ))

    def save(self) -> int:
        """Serialize all layers to disk and return the file size in bytes."""
        with open(self.path, "wb") as f:
            # header
            f.write(MAGIC)
            f.write(struct.pack("<I", len(self._layers)))

            # index section: per-layer metadata
            for key, shape, K, pi, mu, sigma, ic, rc in self._layers:
                nb = key.encode("utf-8")
                f.write(struct.pack("<H", len(nb)))
                f.write(nb)
                f.write(struct.pack("<B", len(shape)))
                for s in shape:
                    f.write(struct.pack("<I", s))
                f.write(struct.pack("<B", K))
                pi.tofile(f)
                mu.tofile(f)
                sigma.tofile(f)
                f.write(struct.pack("<II", len(ic), len(rc)))

            # data section: compressed payloads in index order
            for _, _, _, _, _, _, ic, rc in self._layers:
                ic.tofile(f)
                rc.tofile(f)

        return self.path.stat().st_size

    def summary(self) -> str:
        """Return a human-readable compression summary string."""
        total_orig = sum(int(np.prod(s)) * 4 for _, s, *_ in self._layers)
        total_comp = self.path.stat().st_size
        return (
            f"{len(self._layers)} layers | "
            f"original {total_orig / 1024 / 1024:.1f} MB → "
            f"compressed {total_comp / 1024 / 1024:.1f} MB | "
            f"ratio {total_orig / total_comp:.3f}x"
        )


class GaussReader:
    """Read and decompress layers from a .gauss file.

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
            if magic != MAGIC:
                raise ValueError(f"Invalid .gauss file: magic={magic!r}")
            n = struct.unpack("<I", f.read(4))[0]

            for _ in range(n):
                name_len = struct.unpack("<H", f.read(2))[0]
                key = f.read(name_len).decode("utf-8")
                ndim = struct.unpack("<B", f.read(1))[0]
                shape = tuple(
                    struct.unpack("<I", f.read(4))[0] for _ in range(ndim)
                )
                K = struct.unpack("<B", f.read(1))[0]
                pi = np.frombuffer(f.read(K * 8), np.float64).copy()
                mu = np.frombuffer(f.read(K * 8), np.float64).copy()
                sigma = np.frombuffer(f.read(K * 8), np.float64).copy()
                ic_words, rc_words = struct.unpack("<II", f.read(8))
                self._index.append(dict(
                    key=key,
                    shape=shape,
                    K=K,
                    pi=pi,
                    mu=mu,
                    sigma=sigma,
                    ic_words=ic_words,
                    rc_words=rc_words,
                ))

            self._data_offset = f.tell()

    def keys(self) -> list:
        """Return a list of all layer names stored in this file."""
        return [e["key"] for e in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def decompress(self, key: str) -> np.ndarray:
        """Decompress a single layer and return a float32 numpy array.

        The returned array has the same shape as the original tensor.
        """
        import constriction

        try:
            from .gmm_compress import RESID_BOUND, RESID_SCALE
        except ImportError:
            from gmm_compress import RESID_BOUND, RESID_SCALE  # script mode

        e = next((x for x in self._index if x["key"] == key), None)
        if e is None:
            raise KeyError(key)

        # compute byte offset for this layer within the data section
        offset = self._data_offset
        for x in self._index:
            if x["key"] == key:
                break
            offset += (x["ic_words"] + x["rc_words"]) * 4

        with open(self.path, "rb") as f:
            f.seek(offset)
            ic = np.frombuffer(f.read(e["ic_words"] * 4), np.uint32).copy()
            rc = np.frombuffer(f.read(e["rc_words"] * 4), np.uint32).copy()

        N = int(np.prod(e["shape"]))
        pi, mu, sigma = e["pi"], e["mu"], e["sigma"]

        # decode component assignments
        probs = (pi / pi.sum()).astype(np.float32)
        model = constriction.stream.model.Categorical(probs, perfect=False)
        ans = constriction.stream.stack.AnsCoder(ic)
        asgn = ans.decode(model, N)

        # decode quantized residuals
        sigma_a = sigma[asgn]
        means = np.zeros(N, np.float64)
        stds = np.maximum(sigma_a * RESID_SCALE, 0.5)
        model_g = constriction.stream.model.QuantizedGaussian(
            -RESID_BOUND, RESID_BOUND
        )
        ans2 = constriction.stream.stack.AnsCoder(rc)
        resid_q = ans2.decode(model_g, means, stds)

        weights = (mu[asgn] + resid_q / RESID_SCALE).astype(np.float32)
        return weights.reshape(e["shape"])

    def decompress_all(self) -> dict:
        """Decompress all layers and return a ``{key: np.ndarray}`` dict."""
        return {e["key"]: self.decompress(e["key"]) for e in self._index}


# ---------------------------------------------------------------------------
# CLI: compress / decompress / info
# ---------------------------------------------------------------------------

def _worker_save(task):
    """Compress one layer; return a result dict containing ic/rc arrays."""
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

    key, data_np, K, max_iter, max_fit, _ = task
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
            "key": key, "pi": pi, "mu": mu, "sigma": sigma,
            "ic": ic, "rc": rc, "K": K,
            "orig": orig, "comp": comp,
            "ratio": orig / comp, "n_iter": n_iter,
            "t": time.perf_counter() - t0, "error": None,
        }
    except Exception:
        return {"key": key, "error": traceback.format_exc()}


def main():
    import argparse
    import time
    from multiprocessing import Pool, cpu_count

    import torch
    from safetensors.torch import load_file, save_file

    from .gmm_compress import (
        RESID_BOUND,
        RESID_SCALE,
        assign_all,
        encode_indices,
        encode_residuals,
        fit_gmm_fast,
    )

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
            comp = (e["ic_words"] + e["rc_words"]) * 4
            print(
                f"  {e['key']:<50}  {n * 4 // 1024:>6}KB → "
                f"{comp // 1024:>5}KB  {n * 4 / comp:.3f}x  K={e['K']}"
            )

    elif args.cmd == "compress":
        print(f"Loading {args.input} ...")
        sd = load_file(args.input)
        keys = sorted(
            list(sd.keys()),
            key=lambda k: sd[k].numel(),
            reverse=True,
        )
        n_workers = args.workers or cpu_count()
        tasks = [
            (
                k,
                sd[k].float().numpy().flatten(),
                args.K,
                args.max_iter,
                200_000,
                False,
            )
            for k in keys
        ]
        print(f"Layers: {len(keys)}, K={args.K}, workers={n_workers}")

        shapes = {k: tuple(sd[k].shape) for k in keys}
        total_orig = total_comp = 0
        writer = GaussWriter(args.output)
        t0 = time.perf_counter()

        with Pool(n_workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(_worker_save, tasks), 1
            ):
                if res["error"]:
                    print(f"  ERROR [{res['key']}]")
                    continue
                writer.add(
                    res["key"], shapes[res["key"]], res["K"],
                    res["pi"], res["mu"], res["sigma"],
                    res["ic"], res["rc"],
                )
                total_orig += res["orig"]
                total_comp += res["comp"]
                print(
                    f"  [{i:4d}/{len(tasks)}] {res['key']:<48}"
                    f"  {res['ratio']:.3f}x  iter={res['n_iter']:3d}"
                    f"  {res['t']:.1f}s"
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
