# tqp-reference-plugin: the reference out-of-tree quantizer for turboquant-pro
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""A tiny, pure-NumPy quantizer that exercises the *whole* plugin contract.

This package exists for one reason: to prove that a quantizer living **outside**
the turboquant-pro repository can register through the ``turboquant_pro.plugins``
entry point, pass the executable conformance kit, and be certified by the same
instruments — with no code in the main tree. It is deliberately not novel; it is
the reference the docs point at.

The quantizer is per-channel symmetric uniform quantization on the KV block
convention ``(B=1, H, S, D)`` (and any ``(..., D)`` array for the embedding /
weight tracks):

* per-channel absmax scale over the token axis, a fixed symmetric integer grid
  ``[-qmax, qmax]`` with ``qmax = 2**(bits-1) - 1``;
* the **affine capability** — ``grid_params`` / ``codes`` express the container
  as ``mu + weight * grid[code]`` with ``mu = 0`` and ``weight`` the per-channel
  scale, so it inherits the fused compute-on-codes decode and passes the
  algebraic affine gate exactly;
* **honest bit-packing** — ``compress(packed=True)`` stores the codes ``bits``
  wide (no byte padding) and the container self-reports ``packed``; and
* **serialization** — ``to_bytes`` / ``from_bytes`` round-trip the container.

So every applicable conformance check *passes* (not merely skips): roundtrip,
packed, affine, and serialization. There is no sparse overlay, so the CSR check
skips — outlier decomposition is demonstrated by the in-tree LLM.int8 format.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
from turboquant_pro.plugins import (
    TARGET_EMBEDDING,
    TARGET_KV_KEY,
    TARGET_WEIGHT,
    PluginSpec,
)

__all__ = [
    "ReferenceUniformQuantizer",
    "ReferenceContainer",
    "SPEC",
]


def _bitpack(codes: np.ndarray, width: int) -> np.ndarray:
    """Pack unsigned ``codes`` (< 2**width) into a dense ``width``-bit stream."""
    flat = codes.ravel().astype(np.uint8)
    # MSB-first bit expansion, matching np.packbits' default order.
    bits = ((flat[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)
    return np.packbits(bits.ravel())


def _bitunpack(buf: np.ndarray, n: int, width: int) -> np.ndarray:
    """Invert :func:`_bitpack` back to ``n`` unsigned codes."""
    bits = np.unpackbits(np.asarray(buf, dtype=np.uint8))[: n * width]
    bits = bits.reshape(n, width)
    weights = (1 << np.arange(width - 1, -1, -1)).astype(np.uint32)
    return (bits * weights).sum(axis=1).astype(np.uint8)


@dataclass
class ReferenceContainer:
    """Opaque compressed form. ``codes`` xor ``payload`` is populated depending
    on ``packed``; ``scale`` broadcasts against the code array."""

    scale: np.ndarray  # keepdims per-channel scale, broadcasts to `shape`
    grid: np.ndarray  # (L,) symmetric integer levels, float32
    shape: tuple  # original array shape
    bits: int
    packed: bool  # True -> `payload` holds the bit-packed stream
    n: int  # element count (for unpacking)
    codes: np.ndarray | None = None  # (shape) uint8 grid indices, unpacked
    payload: np.ndarray | None = None  # (bytes,) uint8, packed stream

    def _unpacked_codes(self) -> np.ndarray:
        if not self.packed:
            assert self.codes is not None
            return self.codes
        assert self.payload is not None
        return _bitunpack(self.payload, self.n, self.bits).reshape(self.shape)

    def to_bytes(self) -> bytes:
        meta = json.dumps(
            {
                "shape": list(self.shape),
                "bits": self.bits,
                "packed": self.packed,
                "n": self.n,
            }
        ).encode()
        payload = self.payload if self.packed else self.codes
        buf = io.BytesIO()
        np.savez(
            buf,
            meta=np.frombuffer(meta, dtype=np.uint8),
            scale=self.scale,
            grid=self.grid,
            payload=np.asarray(payload, dtype=np.uint8),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, b: bytes) -> ReferenceContainer:
        with np.load(io.BytesIO(b), allow_pickle=False) as d:
            meta = json.loads(bytes(d["meta"]).decode())
            scale = d["scale"]
            grid = d["grid"]
            payload = d["payload"]
        shape = tuple(meta["shape"])
        packed = bool(meta["packed"])
        return cls(
            scale=scale,
            grid=grid,
            shape=shape,
            bits=int(meta["bits"]),
            packed=packed,
            n=int(meta["n"]),
            codes=None if packed else payload.reshape(shape),
            payload=payload if packed else None,
        )


class ReferenceUniformQuantizer:
    """Per-channel symmetric uniform quantization; the reference plugin."""

    def __init__(self, bits: int = 4, **_ignored):
        if not 2 <= bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        self.bits = int(bits)
        self.qmax = (1 << (bits - 1)) - 1
        # Symmetric integer grid [-qmax, qmax]; code index = level + qmax.
        self.grid = np.arange(-self.qmax, self.qmax + 1, dtype=np.float32)

    # -- core round-trip ------------------------------------------------
    def _scale_axes(self, x: np.ndarray) -> tuple:
        """Axes reduced to form per-channel scale: token axis for the KV block,
        every leading axis otherwise (channel = last axis)."""
        if x.ndim == 4 and x.shape[0] == 1:
            return (2,)  # (1, H, S, D) -> scale over S, keep (1, H, 1, D)
        return tuple(range(x.ndim - 1))

    def compress(self, x: np.ndarray, packed: bool = False) -> ReferenceContainer:
        x = np.asarray(x, dtype=np.float32)
        axes = self._scale_axes(x)
        peak = np.abs(x).max(axis=axes, keepdims=True)
        scale = np.maximum(peak, 1e-12).astype(np.float32) / self.qmax
        idx = np.rint(x / scale).astype(np.int32)
        idx = np.clip(idx, -self.qmax, self.qmax)
        codes = (idx + self.qmax).astype(np.uint8)
        if packed:
            return ReferenceContainer(
                scale=scale,
                grid=self.grid,
                shape=x.shape,
                bits=self.bits,
                packed=True,
                n=int(x.size),
                payload=_bitpack(codes, self.bits),
            )
        return ReferenceContainer(
            scale=scale,
            grid=self.grid,
            shape=x.shape,
            bits=self.bits,
            packed=False,
            n=int(x.size),
            codes=codes,
        )

    def decompress(self, c: ReferenceContainer) -> np.ndarray:
        codes = c._unpacked_codes()
        return (c.grid[codes] * c.scale).astype(np.float32)

    # -- affine capability ----------------------------------------------
    def grid_params(self, c: ReferenceContainer):
        """``(mu (H,D), weight (H,D), grid (L,))`` for the ``(1,H,S,D)`` block;
        ``None`` for other shapes (the documented decompress-then-attend
        degrade). ``mu`` is zero (symmetric); ``weight`` is the per-channel
        scale, so ``mu + weight * grid[code]`` equals ``decompress`` exactly."""
        if len(c.shape) != 4 or c.shape[0] != 1:
            return None
        _, H, _, D = c.shape
        weight = np.asarray(c.scale, dtype=np.float32).reshape(H, D)
        mu = np.zeros((H, D), dtype=np.float32)
        return mu, weight, np.asarray(c.grid, dtype=np.float32)

    def codes(self, c: ReferenceContainer) -> np.ndarray:
        if len(c.shape) != 4:
            raise NotImplementedError("codes() requires the (B,H,S,D) block form")
        return c._unpacked_codes()

    def native_dtype(self):
        return None


def _factory(bits: int = 4, **_ignored) -> ReferenceUniformQuantizer:
    # Accepts and ignores head_dim / n_heads that `tqp plugin conformance` passes
    # to KV plugins, so the CLI's kv_config call path does not error.
    return ReferenceUniformQuantizer(bits=bits)


SPEC = PluginSpec(
    name="tqp_reference",
    factory=_factory,
    targets=frozenset({TARGET_KV_KEY, TARGET_EMBEDDING, TARGET_WEIGHT}),
    tier="experimental",
    description=(
        "Reference out-of-tree plugin: pure-NumPy per-channel symmetric uniform "
        "quantization with the full affine contract, honest bit-packing, and "
        "container serialization — passes roundtrip/packed/affine/serialization "
        "conformance. Exists to prove the plugin ecosystem, not to compete."
    ),
)
