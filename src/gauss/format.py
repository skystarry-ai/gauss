"""
.gauss file format — GMM+ANS compressed weight storage and retrieval.

Format versions
---------------
GAUSS001  Legacy single-shard, no raw section, no dtype field.
GAUSS002  Multi-shard layers, no raw section, no dtype field.
GAUSS003  Adds dtype field, raw section, raw_n_bytes.
GAUSS004  Current format. Adds:
            - resid_scale[2] (uint16 LE) per shard — effective quantization
              scale used when encoding residuals.  Enables adaptive-S for
              reduced-precision dtypes (fp16, bf16).

On-disk layout (GAUSS004)
--------------------------
magic[8]          b"GAUSS004"
n_compressed[4]   uint32 LE — number of GMM-compressed layers
n_raw[4]          uint32 LE — number of raw (lossless) layers

=== Compressed index section (n_compressed entries) ===
name_len[2]       uint16
name[name_len]    UTF-8
ndim[1]           uint8
shape[ndim*4]     uint32 LE per dimension
dtype_id[1]       uint8  (see DTYPE_MAP / DTYPE_ID_MAP)
n_shards[2]       uint16
shard_rows[4]     uint32
  repeated n_shards:
    K[1]            uint8
    pi[K*8]         float64 LE
    mu[K*8]         float64 LE
    sigma[K*8]      float64 LE
    resid_scale[2]  uint16 LE  (NEW in v4; fp32 default = 1000)
    idx_words[4]    uint32
    resid_words[4]  uint32

=== Raw index section (n_raw entries) ===
name_len[2]       uint16
name[name_len]    UTF-8
ndim[1]           uint8
shape[ndim*4]     uint32 LE
dtype_id[1]       uint8
raw_n_bytes[4]    uint32  — byte length of the raw payload

=== Compressed data section (n_compressed layers, same order) ===
  per shard: idx_data[idx_words*4], resid_data[resid_words*4]

=== Raw data section (n_raw layers, same order) ===
  raw_bytes[raw_n_bytes]  — little-endian bytes of the original tensor

Random access
-------------
All idx_words / resid_words / raw_n_bytes are in the index section, so the
byte offset of any tensor can be computed with a linear scan and a single
seek — no need to read the entire data section.

Backward compatibility
----------------------
GAUSS001–003 files are still readable; missing resid_scale fields default
to RESID_SCALE (1000) for lossless decompression of legacy files.
"""

import struct
from pathlib import Path

import numpy as np

from .codec import RESID_SCALE

__all__ = ["GaussWriter", "GaussReader", "DTYPE_MAP", "DTYPE_ID_MAP"]

MAGIC_V1 = b"GAUSS001"
MAGIC_V2 = b"GAUSS002"
MAGIC_V3 = b"GAUSS003"
MAGIC_V4 = b"GAUSS004"

# Mapping from torch dtype string → uint8 id stored in the file.
# torch.float32 is the historical default and maps to id 0.
DTYPE_MAP: dict = {
    "torch.float32": 0,
    "torch.float16": 1,
    "torch.bfloat16": 2,
    "torch.float64": 3,
    "torch.int8": 4,
    "torch.int16": 5,
    "torch.int32": 6,
    "torch.int64": 7,
}

# Reverse mapping: uint8 id → torch dtype string.
DTYPE_ID_MAP: dict = {v: k for k, v in DTYPE_MAP.items()}

# NumPy dtype used for each torch dtype id during raw lossless storage.
_DTYPE_NUMPY: dict = {
    0: np.float32,
    1: np.float16,
    2: np.float32,   # bfloat16 has no numpy equivalent; stored as fp32 bytes
    3: np.float64,
    4: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
}


def _dtype_id(tensor) -> int:
    """Return the uint8 dtype id for a torch tensor."""
    key = str(tensor.dtype)
    return DTYPE_MAP.get(key, 0)


class GaussWriter:
    """Write GMM-compressed and raw-lossless layers to a .gauss file.

    Usage
    -----
    >>> w = GaussWriter("model.gauss")
    >>> # GMM-compressed layer (possibly multi-shard):
    >>> w.add_shards(key, shape, dtype_id, shard_rows, shards)
    >>> # Lossless raw layer (small / 1-D tensors):
    >>> w.add_raw(key, shape, dtype_id, raw_bytes)
    >>> w.save()
    >>> print(w.summary())
    """

    def __init__(self, path: str):
        self.path = Path(path)
        # Compressed entries: (key, shape, dtype_id, shard_rows, shards)
        self._compressed: list = []
        # Raw entries: (key, shape, dtype_id, raw_bytes)
        self._raw: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        dtype_id: int = 0,
    ):
        """Append a single non-sharded compressed layer (convenience wrapper).

        Parameters
        ----------
        key:      Tensor name used as the lookup key.
        shape:    Original tensor shape tuple.
        K:        Number of GMM components.
        pi:       Component weights  (K,) float64.
        mu:       Component means    (K,) float64.
        sigma:    Component std-devs (K,) float64.
        ic:       ANS-encoded component indices (uint32 array).
        rc:       ANS-encoded quantized residuals (uint32 array).
        dtype_id: Torch dtype id (see DTYPE_MAP); default 0 = float32.
        """
        self.add_shards(
            key, shape, dtype_id, 0,
            [{"K": K, "pi": pi, "mu": mu, "sigma": sigma, "ic": ic, "rc": rc}],
        )

    def add_shards(
        self,
        key: str,
        shape: tuple,
        dtype_id: int,
        shard_rows: int,
        shards: list,
    ):
        """Append a (possibly multi-shard) compressed layer.

        Parameters
        ----------
        key:        Tensor name.
        shape:      Original tensor shape.
        dtype_id:   Torch dtype id (see DTYPE_MAP).
        shard_rows: Rows per shard; 0 for non-sharded layers.
        shards:     List of dicts with keys K, pi, mu, sigma, ic, rc.
        """
        normalized = [
            {
                "K": int(s["K"]),
                "pi": np.asarray(s["pi"], np.float64),
                "mu": np.asarray(s["mu"], np.float64),
                "sigma": np.asarray(s["sigma"], np.float64),
                "ic": np.asarray(s["ic"], np.uint32),
                "rc": np.asarray(s["rc"], np.uint32),
            }
            for s in shards
        ]
        self._compressed.append((key, tuple(shape), int(dtype_id),
                                  int(shard_rows), normalized))

    def add_raw(
        self,
        key: str,
        shape: tuple,
        dtype_id: int,
        raw_bytes: bytes,
    ):
        """Store a tensor losslessly in the raw section.

        Intended for small tensors (numel < RAW_THRESHOLD) and 1-D tensors
        such as LayerNorm scale/bias vectors.

        Parameters
        ----------
        key:       Tensor name.
        shape:     Original tensor shape.
        dtype_id:  Torch dtype id (see DTYPE_MAP).
        raw_bytes: Little-endian raw byte representation of the tensor.
        """
        self._raw.append((key, tuple(shape), int(dtype_id), raw_bytes))

    def save(self) -> int:
        """Serialize all layers to disk and return the file size in bytes."""
        with open(self.path, "wb") as f:
            # File header (GAUSS004).
            f.write(MAGIC_V4)
            f.write(struct.pack("<II", len(self._compressed), len(self._raw)))

            # Compressed index section.
            for key, shape, dtype_id, shard_rows, shards in self._compressed:
                _write_name(f, key)
                _write_shape(f, shape)
                f.write(struct.pack("<B", dtype_id))
                f.write(struct.pack("<HI", len(shards), shard_rows))
                for sh in shards:
                    K = sh["K"]
                    # resid_scale defaults to RESID_SCALE for fp32 shards.
                    resid_scale = int(sh.get("resid_scale", RESID_SCALE))
                    f.write(struct.pack("<B", K))
                    sh["pi"].tofile(f)
                    sh["mu"].tofile(f)
                    sh["sigma"].tofile(f)
                    # v4: write resid_scale before word counts.
                    f.write(struct.pack("<H", resid_scale))
                    f.write(struct.pack("<II",
                                       len(sh["ic"]), len(sh["rc"])))

            # Raw index section.
            for key, shape, dtype_id, raw_bytes in self._raw:
                _write_name(f, key)
                _write_shape(f, shape)
                f.write(struct.pack("<B", dtype_id))
                f.write(struct.pack("<I", len(raw_bytes)))

            # Compressed data section (payloads in index order).
            for _, _, _, _, shards in self._compressed:
                for sh in shards:
                    sh["ic"].tofile(f)
                    sh["rc"].tofile(f)

            # Raw data section.
            for _, _, _, raw_bytes in self._raw:
                f.write(raw_bytes)

        return self.path.stat().st_size

    def summary(self) -> str:
        """Return a human-readable compression summary string."""
        # Estimate original size in bytes using the dtype byte-width.
        total_orig = 0
        for key, shape, dtype_id, shard_rows, shards in self._compressed:
            dtype_bytes = np.dtype(_DTYPE_NUMPY[dtype_id]).itemsize
            total_orig += int(np.prod(shape)) * dtype_bytes
        for key, shape, dtype_id, raw_bytes in self._raw:
            total_orig += len(raw_bytes)
        total_comp = self.path.stat().st_size
        n_sharded = sum(
            1 for _, _, _, _, shards in self._compressed if len(shards) > 1
        )
        shard_note = f", {n_sharded} sharded" if n_sharded else ""
        n_raw = len(self._raw)
        raw_note = f", {n_raw} raw" if n_raw else ""
        return (
            f"{len(self._compressed)} compressed{shard_note}"
            f"{raw_note} | "
            f"original {total_orig / 1024 / 1024:.1f} MB → "
            f"compressed {total_comp / 1024 / 1024:.1f} MB | "
            f"ratio {total_orig / total_comp:.3f}x"
        )


class GaussReader:
    """Read and decompress layers from a .gauss file.

    Supports GAUSS001 (legacy), GAUSS002 (sharded), and GAUSS003 (current,
    with raw section and dtype preservation).

    Usage
    -----
    >>> r = GaussReader("model.gauss")
    >>> print(r.keys())
    >>> arr = r.decompress("model.layers.0.mlp.gate_proj.weight")
    >>> state_dict = r.decompress_all()
    """

    def __init__(self, path: str):
        self.path = Path(path)
        # Compressed entries parsed from the index section.
        self._index: list = []
        # Raw entries parsed from the raw index section.
        self._raw_index: list = []
        self._data_offset: int = 0
        self._raw_data_offset: int = 0
        self._version: int = 0
        self._parse_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def keys(self) -> list:
        """Return names of all layers (compressed + raw) in this file."""
        comp_keys = [e["key"] for e in self._index]
        raw_keys = [e["key"] for e in self._raw_index]
        return comp_keys + raw_keys

    def __len__(self) -> int:
        return len(self._index) + len(self._raw_index)

    def decompress(self, key: str) -> np.ndarray:
        """Decompress a single layer and return a numpy array.

        The array dtype matches the original tensor dtype recorded at
        compression time (float32 for GAUSS001/002 legacy files).

        For sharded layers all shards are decoded and concatenated before
        reshaping to the original tensor shape.
        """
        # Check raw section first.
        raw_entry = next(
            (x for x in self._raw_index if x["key"] == key), None
        )
        if raw_entry is not None:
            return self._decompress_raw(raw_entry)

        entry = next((x for x in self._index if x["key"] == key), None)
        if entry is None:
            raise KeyError(key)
        return self._decompress_gmm(entry)

    def decompress_all(self) -> dict:
        """Decompress all layers and return a ``{key: np.ndarray}`` dict."""
        result = {}
        for e in self._index:
            result[e["key"]] = self._decompress_gmm(e)
        for e in self._raw_index:
            result[e["key"]] = self._decompress_raw(e)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_index(self):
        """Parse the index section(s) from the file header."""
        with open(self.path, "rb") as f:
            magic = f.read(8)
            if magic == MAGIC_V1:
                self._version = 1
            elif magic == MAGIC_V2:
                self._version = 2
            elif magic == MAGIC_V3:
                self._version = 3
            elif magic == MAGIC_V4:
                self._version = 4
            else:
                raise ValueError(
                    f"Unrecognised .gauss file magic: {magic!r}"
                )

            if self._version >= 3:
                n_comp, n_raw = struct.unpack("<II", f.read(8))
            else:
                n_comp = struct.unpack("<I", f.read(4))[0]
                n_raw = 0

            # Parse compressed index.
            for _ in range(n_comp):
                key = _read_name(f)
                shape = _read_shape(f)

                if self._version == 1:
                    dtype_id = 0  # legacy: always float32
                    K = struct.unpack("<B", f.read(1))[0]
                    pi = _read_f64(f, K)
                    mu = _read_f64(f, K)
                    sigma = _read_f64(f, K)
                    ic_words, rc_words = struct.unpack("<II", f.read(8))
                    shards = [dict(K=K, pi=pi, mu=mu, sigma=sigma,
                                   ic_words=ic_words, rc_words=rc_words,
                                   resid_scale=RESID_SCALE)]
                    shard_rows = 0
                elif self._version == 2:
                    dtype_id = 0  # legacy: always float32
                    n_shards, shard_rows = struct.unpack("<HI", f.read(6))
                    shards = _read_shards(f, n_shards, read_resid_scale=False)
                elif self._version == 3:
                    dtype_id = struct.unpack("<B", f.read(1))[0]
                    n_shards, shard_rows = struct.unpack("<HI", f.read(6))
                    shards = _read_shards(f, n_shards, read_resid_scale=False)
                else:  # v4
                    dtype_id = struct.unpack("<B", f.read(1))[0]
                    n_shards, shard_rows = struct.unpack("<HI", f.read(6))
                    shards = _read_shards(f, n_shards, read_resid_scale=True)

                self._index.append(dict(
                    key=key, shape=shape, dtype_id=dtype_id,
                    shard_rows=shard_rows, shards=shards,
                ))

            # Parse raw index (v3+ only).
            for _ in range(n_raw):
                key = _read_name(f)
                shape = _read_shape(f)
                dtype_id = struct.unpack("<B", f.read(1))[0]
                raw_n_bytes = struct.unpack("<I", f.read(4))[0]
                self._raw_index.append(dict(
                    key=key, shape=shape, dtype_id=dtype_id,
                    raw_n_bytes=raw_n_bytes,
                ))

            self._data_offset = f.tell()

        # Compute raw data section start by skipping the compressed payloads.
        offset = self._data_offset
        for entry in self._index:
            for sh in entry["shards"]:
                offset += (sh["ic_words"] + sh["rc_words"]) * 4
        self._raw_data_offset = offset

    def _decompress_gmm(self, entry: dict) -> np.ndarray:
        """Decode one GMM-compressed entry and return a numpy array."""
        from .codec import RESID_BOUND, RESID_SCALE
        import constriction

        shape = entry["shape"]
        dtype_id = entry["dtype_id"]
        shard_rows = entry["shard_rows"]
        shards = entry["shards"]
        n_shards = len(shards)
        total_elements = int(np.prod(shape))

        # Compute byte offset for this entry's data.
        offset = self._data_offset
        for x in self._index:
            if x["key"] == entry["key"]:
                break
            for sh in x["shards"]:
                offset += (sh["ic_words"] + sh["rc_words"]) * 4

        # Per-shard element counts.
        if n_shards == 1:
            shard_n = [total_elements]
        else:
            row_size = int(np.prod(shape[1:]))
            shard_n = [shard_rows * row_size] * (n_shards - 1)
            shard_n.append(
                total_elements - (n_shards - 1) * shard_rows * row_size
            )

        out_chunks = []
        with open(self.path, "rb") as f:
            f.seek(offset)
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
                # Reconstruct in float64, then cast to original dtype.
                out_chunks.append(mu[asgn] + resid_q / s)

        flat = np.concatenate(out_chunks)
        target_np = _DTYPE_NUMPY[dtype_id]
        return flat.astype(target_np).reshape(shape)

    def _decompress_raw(self, entry: dict) -> np.ndarray:
        """Read one lossless raw entry and return a numpy array."""
        dtype_id = entry["dtype_id"]
        shape = entry["shape"]

        # Compute byte offset inside the raw data section.
        offset = self._raw_data_offset
        for x in self._raw_index:
            if x["key"] == entry["key"]:
                break
            offset += x["raw_n_bytes"]

        with open(self.path, "rb") as f:
            f.seek(offset)
            raw = f.read(entry["raw_n_bytes"])

        np_dtype = _DTYPE_NUMPY[dtype_id]
        return np.frombuffer(raw, dtype=np_dtype).copy().reshape(shape)


# ---------------------------------------------------------------------------
# Private I/O helpers
# ---------------------------------------------------------------------------

def _write_name(f, name: str):
    nb = name.encode("utf-8")
    f.write(struct.pack("<H", len(nb)))
    f.write(nb)


def _write_shape(f, shape: tuple):
    f.write(struct.pack("<B", len(shape)))
    for s in shape:
        f.write(struct.pack("<I", s))


def _read_name(f) -> str:
    (n,) = struct.unpack("<H", f.read(2))
    return f.read(n).decode("utf-8")


def _read_shape(f) -> tuple:
    (ndim,) = struct.unpack("<B", f.read(1))
    return tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))


def _read_f64(f, K: int) -> np.ndarray:
    return np.frombuffer(f.read(K * 8), np.float64).copy()


def _read_shards(f, n_shards: int, read_resid_scale: bool = False) -> list:
    """Read n_shards shard descriptors from the index section.

    Parameters
    ----------
    f:                Open file positioned at the first shard descriptor.
    n_shards:         Number of shards to read.
    read_resid_scale: True for GAUSS004+ files which store a uint16
                      resid_scale field after the GMM parameters.
                      False for GAUSS001–003 (legacy); scale defaults to
                      RESID_SCALE (1000).
    """
    from .codec import RESID_SCALE as _DEFAULT_SCALE

    shards = []
    for _ in range(n_shards):
        K = struct.unpack("<B", f.read(1))[0]
        pi = _read_f64(f, K)
        mu = _read_f64(f, K)
        sigma = _read_f64(f, K)
        if read_resid_scale:
            (resid_scale,) = struct.unpack("<H", f.read(2))
        else:
            resid_scale = _DEFAULT_SCALE
        ic_words, rc_words = struct.unpack("<II", f.read(8))
        shards.append(dict(K=K, pi=pi, mu=mu, sigma=sigma,
                           resid_scale=resid_scale,
                           ic_words=ic_words, rc_words=rc_words))
    return shards
