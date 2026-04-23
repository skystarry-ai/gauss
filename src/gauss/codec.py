"""
ANS codec wrappers (constriction library) and quantization constants.

RESID_SCALE controls the float-to-integer quantization resolution.
The maximum per-element reconstruction error is bounded by ±1/(2*RESID_SCALE).
"""

import numpy as np

__all__ = [
    "RESID_SCALE",
    "RESID_BOUND",
    "encode_indices",
    "decode_indices",
    "encode_residuals",
    "decode_residuals",
]

# Quantization scale: float residual → integer.
# Reconstruction error is bounded by ±1/(2 * RESID_SCALE) = ±5e-4.
RESID_SCALE: int = 1000

# Symmetric clamp range for quantized residuals (INT16 max).
RESID_BOUND: int = 32767


def encode_indices(asgn: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """ANS-encode component assignment indices using a Categorical model.

    Parameters
    ----------
    asgn: Integer assignment array (N,) dtype int32.
    pi:   Component weights (K,) float64; will be L1-normalized internally.

    Returns
    -------
    Compressed uint32 array (ANS bitstream).
    """
    import constriction

    ans = constriction.stream.stack.AnsCoder()
    probs = (pi / pi.sum()).astype(np.float32)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    ans.encode_reverse(asgn.astype(np.int32), model)
    return ans.get_compressed()


def decode_indices(
    compressed: np.ndarray,
    pi: np.ndarray,
    N: int,
) -> np.ndarray:
    """ANS-decode component assignment indices.

    Parameters
    ----------
    compressed: uint32 ANS bitstream from ``encode_indices``.
    pi:         Component weights (K,) float64.
    N:          Number of elements to decode.

    Returns
    -------
    Integer assignment array (N,) dtype int32.
    """
    import constriction

    probs = (pi / pi.sum()).astype(np.float32)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    ans = constriction.stream.stack.AnsCoder(compressed)
    return ans.decode(model, N)


def encode_residuals(
    resid_q: np.ndarray,
    sigma_a: np.ndarray,
) -> np.ndarray:
    """ANS-encode quantized residuals with per-element Gaussian models.

    Each residual is modeled as N(0, (sigma_a * RESID_SCALE)^2) and coded
    with a QuantizedGaussian model, approaching the Shannon entropy
    bound.

    Parameters
    ----------
    resid_q: Quantized residuals (N,) int32, clipped to ±RESID_BOUND.
    sigma_a: Per-element component std-devs (N,) float64.

    Returns
    -------
    Compressed uint32 array (ANS bitstream).
    """
    import constriction

    ans = constriction.stream.stack.AnsCoder()
    model = constriction.stream.model.QuantizedGaussian(
        -RESID_BOUND, RESID_BOUND
    )
    means = np.zeros(len(resid_q), dtype=np.float64)
    stds = np.maximum(sigma_a * RESID_SCALE, 0.5).astype(np.float64)
    ans.encode_reverse(resid_q.astype(np.int32), model, means, stds)
    return ans.get_compressed()


def decode_residuals(
    compressed: np.ndarray,
    sigma_a: np.ndarray,
    N: int,
) -> np.ndarray:
    """ANS-decode quantized residuals.

    Parameters
    ----------
    compressed: uint32 ANS bitstream from ``encode_residuals``.
    sigma_a:    Per-element component std-devs (N,) float64.
    N:          Number of elements to decode.

    Returns
    -------
    Decoded quantized residuals (N,) int32.
    """
    import constriction

    ans = constriction.stream.stack.AnsCoder(compressed)
    model = constriction.stream.model.QuantizedGaussian(
        -RESID_BOUND, RESID_BOUND
    )
    means = np.zeros(N, dtype=np.float64)
    stds = np.maximum(sigma_a * RESID_SCALE, 0.5).astype(np.float64)
    return ans.decode(model, means, stds)
