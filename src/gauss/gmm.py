"""
GMM fitting and chunk-wise assignment.

Pure NumPy hard EM — no scikit-learn dependency, roughly 3-5x faster than
soft EM for compression workloads where we only need cluster assignments.
"""

import numpy as np

__all__ = ["fit_gmm_fast", "assign_all"]


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
    data:     1-D float64 array of weight values to fit.
    K:        Number of Gaussian components.
    max_iter: Maximum number of EM iterations.
    tol:      Unused; kept for API compatibility.

    Returns
    -------
    pi, mu, sigma : Component weights, means, std-devs — each shape (K,).
    n_iter        : Number of iterations executed before stopping.
    """
    data = data.astype(np.float64)
    N = len(data)
    d2 = data ** 2  # precomputed to speed up M-step variance calculation

    # Deterministic init: evenly-spaced percentiles across the value range.
    mu = np.percentile(data, np.linspace(5, 95, K))
    sigma = np.full(K, max(data.std() / K, 1e-6))
    pi = np.ones(K) / K
    prev_asgn = None

    for i in range(max_iter):
        # E-step: log-responsibilities, take argmax (hard assignment).
        inv_s = 1.0 / (sigma + 1e-10)
        diff = data[:, None] - mu[None, :]
        log_r = (
            np.log(pi + 1e-300)
            - np.log(sigma + 1e-10)
            - 0.5 * (diff * inv_s[None, :]) ** 2
        )
        asgn = log_r.argmax(axis=1).astype(np.int32)

        # Hard EM rarely converges to zero change; 0.1% threshold is used.
        if prev_asgn is not None:
            if np.mean(asgn != prev_asgn) < 0.001:
                return pi, mu, sigma, i + 1
        prev_asgn = asgn

        # M-step: fully vectorized with bincount (no Python loop over K).
        counts = np.bincount(asgn, minlength=K).astype(np.float64)
        pi = np.where(counts > 0, counts / N, 0.0)

        # Guard against empty clusters by falling back to previous parameters.
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


def assign_all(
    data: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    chunk: int = 500_000,
) -> np.ndarray:
    """Assign every data point to its nearest GMM component.

    Processes data in chunks to bound peak memory to ``chunk × K × 8`` bytes.

    Parameters
    ----------
    data:  1-D float64 array of weight values.
    pi:    Component weights  (K,) float64.
    mu:    Component means    (K,) float64.
    sigma: Component std-devs (K,) float64.
    chunk: Number of elements processed per iteration.

    Returns
    -------
    Integer assignment array of shape (N,), dtype int32.
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
