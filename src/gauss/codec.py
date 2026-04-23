"""
ANS codec wrappers (constriction library) and quantization constants.

RESID_SCALE controls the float-to-integer quantization resolution.
The maximum per-element reconstruction error is bounded by ±1/(2*RESID_SCALE).

Adaptive-S
----------
For reduced-precision dtypes (fp16, bf16) the native machine epsilon already
limits meaningful residual precision.  Pushing RESID_SCALE beyond

    S_adaptive = min(S_target, 1 / (2 * precision_limit))

where ``precision_limit = eps / 2`` (half the ULP at 1.0) would waste ANS
bitstream capacity encoding sub-epsilon noise.  Use ``adaptive_resid_scale``
to obtain the dtype-appropriate scale before building quantized residuals.
"""

import numpy as np

__all__ = [
    "RESID_SCALE",
    "RESID_BOUND",
    # Adaptive-S API
    "DTYPE_PRECISION_LIMITS",
    "adaptive_resid_scale",
    # Codec functions
    "encode_indices",
    "decode_indices",
    "encode_residuals",
    "decode_residuals",
]

# Quantization scale: float residual → integer.
# Reconstruction error is bounded by ±1/(2 * RESID_SCALE) = ±5e-4.
# This is the default (fp32) target scale.
RESID_SCALE: int = 1000

# Symmetric clamp range for quantized residuals (INT16 max).
RESID_BOUND: int = 32767

# ---------------------------------------------------------------------------
# Adaptive-S: per-dtype precision limits
# ---------------------------------------------------------------------------

# Half the ULP at 1.0 for each supported floating-point dtype.
# precision_limit = np.finfo(dtype).eps / 2
# This is the smallest interval that the dtype can reliably distinguish.
DTYPE_PRECISION_LIMITS: dict = {
    "float32": float(np.finfo(np.float32).eps) / 2.0,   # ~5.96e-8
    "float16": float(np.finfo(np.float16).eps) / 2.0,   # ~4.88e-4
    # bfloat16 has no numpy equivalent; use the known IEEE definition.
    # eps(bf16) = 2^-7 ≈ 7.8125e-3  →  precision_limit ≈ 3.906e-3
    "bfloat16": 2.0 ** -8,                               # ~3.906e-3
}


def adaptive_resid_scale(dtype_str: str, s_target: int = RESID_SCALE) -> int:
    """Return the effective residual quantization scale for *dtype_str*.

    Applies the adaptive-S formula::

        S_adaptive = min(S_target, floor(1 / (2 * precision_limit)))

    For fp32 the upper bound (~8.4e6) is always above any reasonable
    S_target, so the function simply returns S_target unchanged.
    For fp16 the upper bound is ~1024 and for bf16 it is ~128, capping
    RESID_SCALE automatically when the dtype cannot represent finer
    residuals anyway.

    Parameters
    ----------
    dtype_str:
        Dtype name as a plain string: ``"float32"``, ``"float16"``, or
        ``"bfloat16"``.  Torch-style prefixes (``"torch.float16"``) are
        stripped automatically.
    s_target:
        Desired quantization scale before the precision cap is applied.
        Defaults to the module-level ``RESID_SCALE``.

    Returns
    -------
    int
        The capped scale value, always ≥ 1.
    """
    # Normalise "torch.float16" → "float16", etc.
    key = dtype_str.split(".")[-1]

    limit = DTYPE_PRECISION_LIMITS.get(key)
    if limit is None:
        # Unknown dtype — fall back to the target scale unchanged.
        return s_target

    # S_adaptive = min(S_target, floor(1 / (2 * precision_limit)))
    s_cap = int(1.0 / (2.0 * limit))
    return min(s_target, s_cap)


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
    scale: int = RESID_SCALE,
) -> np.ndarray:
    """ANS-encode quantized residuals with per-element Gaussian models.

    Each residual is modeled as N(0, (sigma_a * scale)^2) and coded with a
    QuantizedGaussian model, approaching the Shannon entropy bound.

    Parameters
    ----------
    resid_q: Quantized residuals (N,) int32, clipped to ±RESID_BOUND.
    sigma_a: Per-element component std-devs (N,) float64.
    scale:   Quantization scale used when building resid_q.  Defaults to the
             module-level RESID_SCALE; pass ``adaptive_resid_scale(dtype)``
             when compressing reduced-precision tensors.

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
    stds = np.maximum(sigma_a * scale, 0.5).astype(np.float64)
    ans.encode_reverse(resid_q.astype(np.int32), model, means, stds)
    return ans.get_compressed()


def decode_residuals(
    compressed: np.ndarray,
    sigma_a: np.ndarray,
    N: int,
    scale: int = RESID_SCALE,
) -> np.ndarray:
    """ANS-decode quantized residuals.

    Parameters
    ----------
    compressed: uint32 ANS bitstream from ``encode_residuals``.
    sigma_a:    Per-element component std-devs (N,) float64.
    N:          Number of elements to decode.
    scale:      Must match the ``scale`` value used during encoding.

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
    stds = np.maximum(sigma_a * scale, 0.5).astype(np.float64)
    return ans.decode(model, means, stds)
