"""
GMM + ANS weight compression.

Uses numpy-only hard EM for Gaussian Mixture Model fitting — no sklearn
dependency, roughly 3–5× faster than soft EM for this use case.

Key design choices
------------------
- Layers sorted largest-first for better worker load balancing.
- Single process pool; no separate profiling phase.
- Optional ``--verify`` flag to check reconstruction error after encoding.
"""

import time
import traceback
from multiprocessing import Pool, cpu_count

import constriction
import numpy as np
import torch

RESID_SCALE = 1000   # quantization scale: float residual → integer
RESID_BOUND = 32767  # symmetric clamp range for quantized residuals


def fit_gmm_fast(
    data: np.ndarray,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-5,
):
    """Fit a K-component GMM using numpy-only hard EM.

    The M-step is fully vectorized via ``np.bincount``; there are no Python
    loops over components.  Early stopping triggers when fewer than 0.1% of
    assignments change between consecutive iterations.

    Parameters
    ----------
    data:     1-D float64 array of weight values to fit
    K:        number of Gaussian components
    max_iter: maximum number of EM iterations
    tol:      unused; kept for API compatibility

    Returns
    -------
    pi, mu, sigma : component weights, means, std devs — each shape (K,)
    n_iter        : number of iterations executed before stopping
    """
    data = data.astype(np.float64)
    N = len(data)
    d2 = data ** 2  # precomputed to speed up M-step variance calculation

    # deterministic initialization via evenly-spaced percentiles
    mu = np.percentile(data, np.linspace(5, 95, K))
    sigma = np.full(K, max(data.std() / K, 1e-6))
    pi = np.ones(K) / K
    prev_asgn = None

    for i in range(max_iter):
        # E-step: compute log-responsibilities and take argmax (hard assignment)
        inv_s = 1.0 / (sigma + 1e-10)
        diff = data[:, None] - mu[None, :]
        log_r = (
            np.log(pi + 1e-300)
            - np.log(sigma + 1e-10)
            - 0.5 * (diff * inv_s[None, :]) ** 2
        )
        asgn = log_r.argmax(axis=1).astype(np.int32)

        # hard EM rarely converges to zero change, so 0.1% threshold is used
        if prev_asgn is not None:
            changed = np.mean(asgn != prev_asgn)
            if changed < 0.001:
                return pi, mu, sigma, i + 1
        prev_asgn = asgn

        # M-step: fully vectorized with bincount (no Python loop over K)
        counts = np.bincount(asgn, minlength=K).astype(np.float64)
        pi = np.where(counts > 0, counts / N, 0.0)

        # guard against empty clusters by falling back to previous parameters
        c_safe = np.where(counts > 0, counts, 1.0)
        sum_x = np.bincount(asgn, weights=data, minlength=K)
        sum_x2 = np.bincount(asgn, weights=d2, minlength=K)
        new_mu = np.where(counts > 0, sum_x / c_safe, mu)
        var = np.maximum(sum_x2 / c_safe - new_mu ** 2, 0.0)
        new_sigma = np.where(
            counts > 0, np.maximum(np.sqrt(var), 1e-6), sigma
        )

        mu, sigma = new_mu, new_sigma

    return pi, mu, sigma, max_iter


def assign_all(data: np.ndarray, pi, mu, sigma, chunk: int = 500_000):
    """Assign every data point to its nearest GMM component.

    Processes data in chunks to bound peak memory to ``chunk × K × 8`` bytes.
    """
    N = len(data)
    out = np.empty(N, dtype=np.int32)
    log_pi = np.log(pi + 1e-300)
    log_sigma = np.log(sigma + 1e-10)
    inv_sigma = 1.0 / (sigma + 1e-10)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        diff = data[s:e, None] - mu[None, :]
        log_r = log_pi - log_sigma - 0.5 * (diff * inv_sigma[None, :]) ** 2
        out[s:e] = log_r.argmax(axis=1)
    return out


# ---------------------------------------------------------------------------
# ANS codec wrappers (constriction library)
# ---------------------------------------------------------------------------

def encode_indices(asgn, pi):
    """ANS-encode component assignment indices using a Categorical model."""
    ans = constriction.stream.stack.AnsCoder()
    probs = (pi / pi.sum()).astype(np.float32)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    ans.encode_reverse(asgn.astype(np.int32), model)
    return ans.get_compressed()


def decode_indices(compressed, pi, N):
    """ANS-decode component assignment indices."""
    probs = (pi / pi.sum()).astype(np.float32)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    ans = constriction.stream.stack.AnsCoder(compressed)
    return ans.decode(model, N)


def encode_residuals(resid_q, sigma_a):
    """ANS-encode quantized residuals using per-component Gaussian models."""
    ans = constriction.stream.stack.AnsCoder()
    model = constriction.stream.model.QuantizedGaussian(
        -RESID_BOUND, RESID_BOUND
    )
    means = np.zeros(len(resid_q), dtype=np.float64)
    stds = np.maximum(sigma_a * RESID_SCALE, 0.5).astype(np.float64)
    ans.encode_reverse(resid_q.astype(np.int32), model, means, stds)
    return ans.get_compressed()


def decode_residuals(compressed, sigma_a, N):
    """ANS-decode quantized residuals."""
    ans = constriction.stream.stack.AnsCoder(compressed)
    model = constriction.stream.model.QuantizedGaussian(
        -RESID_BOUND, RESID_BOUND
    )
    means = np.zeros(N, dtype=np.float64)
    stds = np.maximum(sigma_a * RESID_SCALE, 0.5).astype(np.float64)
    return ans.decode(model, means, stds)


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compress a single layer; designed to run inside a pool worker process.

    Parameters
    ----------
    args: tuple of (key, data_np, K, max_iter, max_fit_samples, verify)

    Returns
    -------
    dict with keys: key, K, orig, comp, ratio, bits, max_err,
                    t_comp, n_iter, error
    """
    key, data_np, K, max_iter, max_fit_samples, verify = args
    try:
        data = data_np.astype(np.float64)
        N = len(data)
        t0 = time.perf_counter()

        # subsample for GMM fitting to keep memory and runtime bounded
        if N > max_fit_samples:
            idx = np.random.choice(N, max_fit_samples, replace=False)
            fit_data = data[idx]
        else:
            fit_data = data

        pi, mu, sigma, n_iter = fit_gmm_fast(fit_data, K, max_iter)

        # assign every point and compute quantized residuals
        asgn = assign_all(data, pi, mu, sigma)
        mu_a = mu[asgn]
        sigma_a = sigma[asgn]
        resid_q = np.clip(
            np.round((data - mu_a) * RESID_SCALE).astype(np.int32),
            -RESID_BOUND,
            RESID_BOUND,
        )

        # ANS encode
        ic = encode_indices(asgn, pi)
        rc = encode_residuals(resid_q, sigma_a)
        t_comp = time.perf_counter() - t0

        orig_bytes = N * 4
        gmm_meta = K * 3 * 8  # pi + mu + sigma stored as float64
        comp_bytes = (len(ic) + len(rc)) * 4 + gmm_meta

        max_err = None
        if verify:
            asgn2 = decode_indices(ic, pi, N)
            rq2 = decode_residuals(rc, sigma[asgn2], N)
            restored = mu[asgn2] + rq2 / RESID_SCALE
            max_err = float(np.abs(data - restored).max())

        return {
            "key": key, "K": K,
            "orig": orig_bytes, "comp": comp_bytes,
            "ratio": orig_bytes / comp_bytes,
            "bits": (comp_bytes * 8) / N,
            "max_err": max_err,
            "t_comp": t_comp,
            "n_iter": n_iter,
            "error": None,
        }
    except Exception:
        return {
            "key": key, "K": K,
            "error": traceback.format_exc(), "n_iter": None,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pathlib

    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(
        description="GMM+ANS weight compression — numpy hard EM, no sklearn"
    )
    parser.add_argument("safetensors")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-fit-samples", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--layers", type=str, default=None,
        help="comma-separated layer names (all eligible layers if omitted)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="verify reconstruction error after encoding (slower)",
    )
    args = parser.parse_args()

    path = pathlib.Path(args.safetensors)
    if not path.exists():
        print(f"File not found: {path}")
        exit(1)

    print(f"Loading {path} ...")
    sd = load_file(str(path))

    if args.layers:
        keys = [k.strip() for k in args.layers.split(",") if k.strip() in sd]
    else:
        keys = [
            k for k, v in sd.items()
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and v.numel() >= 1000
        ]

    # sort largest-first so workers stay busy until the very end
    keys.sort(key=lambda k: sd[k].numel(), reverse=True)

    n_workers = args.workers or cpu_count()
    tasks = [
        (
            k,
            sd[k].float().numpy().flatten(),
            args.K,
            args.max_iter,
            args.max_fit_samples,
            args.verify,
        )
        for k in keys
    ]

    print(
        f"Layers: {len(keys)}  |  K={args.K}  |  "
        f"workers={n_workers}  |  max_iter={args.max_iter}"
    )
    hdr = f"{'layer':<52} {'orig':>7} {'comp':>6} {'ratio':>7} "
    hdr += f"{'max_err':>10} " if args.verify else " " * 12
    hdr += f"{'iter':>5} {'time':>6}"
    print(hdr)
    print("-" * len(hdr))

    total_orig = total_comp = 0
    results = []

    with Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results.append(res)
            if res["error"]:
                print(f"  ERROR [{res['key']}]:\n{res['error'][:200]}")
                continue
            total_orig += res["orig"]
            total_comp += res["comp"]
            err_str = f"  {res['max_err']:>10.2e}" if args.verify else ""
            print(
                f"  [{i:4d}/{len(tasks)}] {res['key']:<50}"
                f"  {res['orig'] // 1024:>5}KB {res['comp'] // 1024:>5}KB"
                f"  {res['ratio']:>6.3f}x"
                f"{err_str}"
                f"  {res['n_iter']:>5}  {res['t_comp']:>5.1f}s"
            )

    print("=" * len(hdr))
    if total_orig > 0:
        ok = [r for r in results if not r["error"]]
        avg_bits = sum(r["bits"] for r in ok) / len(ok)
        print(
            f"  {'total':<52}"
            f"  {total_orig // 1024 // 1024:>4}MB"
            f" {total_comp // 1024 // 1024:>4}MB"
            f"  {total_orig / total_comp:>6.3f}x"
            f"  avg {avg_bits:.2f} bits/w"
        )
