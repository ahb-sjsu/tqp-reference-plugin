"""The reference plugin passes the executable container-contract conformance kit.

Run in this package's own CI — needs only ``turboquant-pro`` + ``numpy``. The
point of Phase 5 is that *this* runs green from outside the main repo.
"""

from __future__ import annotations

import numpy as np
import pytest
from turboquant_pro.plugin_conformance import assert_conformance, run_conformance
from turboquant_pro.plugins import Quantizer

from tqp_reference.plugin import ReferenceContainer, ReferenceUniformQuantizer

RNG = np.random.default_rng(7)


def _kv_block(h=4, s=96, d=64):
    # per-head DC offset + noise, the canonical KV block the kit builds
    off = RNG.uniform(-4.0, 4.0, size=(1, h, 1, d))
    return (off + RNG.standard_normal((1, h, s, d))).astype(np.float32)


def test_satisfies_quantizer_protocol():
    assert isinstance(ReferenceUniformQuantizer(), Quantizer)


def test_every_applicable_check_passes():
    """roundtrip + packed + affine + serialization PASS; csr skips (no overlay)."""
    q = ReferenceUniformQuantizer(bits=4)
    report = assert_conformance(q, _kv_block())
    assert report.results["roundtrip"].startswith("pass"), report
    assert report.results["packed"].startswith("pass"), report
    assert report.results["affine"].startswith("pass"), report
    assert report.results["serialization"].startswith("pass"), report
    assert report.results["csr"].startswith("skip"), report


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_conformance_across_bit_widths(bits):
    report = run_conformance(ReferenceUniformQuantizer(bits=bits), _kv_block())
    assert report.passed, report


def test_affine_reconstruction_is_exact():
    """The fused-decode safety gate: mu + weight*grid[codes] == decompress."""
    q = ReferenceUniformQuantizer(bits=4)
    c = q.compress(_kv_block())
    mu, weight, grid = q.grid_params(c)
    _, H, S, D = c.shape
    assert mu.shape == (H, D) and weight.shape == (H, D)
    dense = mu[:, None, :] + weight[:, None, :] * grid[q.codes(c)[0]]
    assert np.allclose(dense[None], q.decompress(c), atol=1e-6)


def test_packing_is_honest_and_lossless():
    """packed=True is a distinct, smaller representation that decompresses equally."""
    q = ReferenceUniformQuantizer(bits=4)
    x = _kv_block()
    unpacked = q.compress(x, packed=False)
    packed = q.compress(x, packed=True)
    assert unpacked.packed is False and packed.packed is True
    assert packed.codes is None and packed.payload is not None
    # 4-bit codes -> ~half a byte each; the packed stream is materially smaller.
    assert packed.payload.nbytes < unpacked.codes.nbytes
    assert np.allclose(q.decompress(packed), q.decompress(unpacked), atol=1e-6)


def test_serialization_roundtrip_both_modes():
    q = ReferenceUniformQuantizer(bits=5)
    x = _kv_block()
    for packed in (False, True):
        c = q.compress(x, packed=packed)
        c2 = ReferenceContainer.from_bytes(c.to_bytes())
        assert np.allclose(q.decompress(c2), q.decompress(c), atol=1e-6)
        assert c2.packed is packed


def test_embedding_shape_roundtrips_and_degrades_affine():
    """2-D corpora work for the embedding track; affine correctly returns None."""
    q = ReferenceUniformQuantizer(bits=4)
    x = RNG.standard_normal((256, 100)).astype(np.float32)
    c = q.compress(x)
    recon = q.decompress(c)
    assert recon.shape == x.shape and np.all(np.isfinite(recon))
    assert q.grid_params(c) is None


def test_invalid_bits_rejected():
    with pytest.raises(ValueError):
        ReferenceUniformQuantizer(bits=1)
    with pytest.raises(ValueError):
        ReferenceUniformQuantizer(bits=9)
