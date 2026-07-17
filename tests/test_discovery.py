"""Out-of-tree discovery + certification — the Phase-5 exit criterion itself.

These tests deliberately do **not** import ``tqp_reference`` to make the plugin
available: they prove turboquant-pro finds it through the installed
``turboquant_pro.plugins`` entry point, instantiates it by name, and certifies
its output with the same rank-certificate instrument the whole toolkit uses.
Requires the package to be *installed* (``pip install .``) so the entry point is
registered.
"""

from __future__ import annotations

import numpy as np
from turboquant_pro import (
    available_plugins,
    certificate_from_embeddings,
    create_quantizer,
)
from turboquant_pro.plugins import load_entry_point_plugins

PLUGIN = "tqp_reference"


def test_entry_point_discovery_finds_plugin():
    """Discovered via the entry-point group, without a direct import."""
    load_entry_point_plugins(force=True)
    specs = available_plugins()
    assert PLUGIN in specs, sorted(specs)
    spec = specs[PLUGIN]
    assert spec.tier == "experimental"
    assert "embedding" in spec.targets and "kv_key" in spec.targets


def test_create_by_name_returns_working_quantizer():
    q = create_quantizer(PLUGIN, bits=4)
    x = np.random.default_rng(0).standard_normal((1, 4, 64, 32)).astype(np.float32)
    recon = q.decompress(q.compress(x))
    assert recon.shape == x.shape and np.all(np.isfinite(recon))


def test_participates_in_certification():
    """`tqp certify`'s instrument certifies the plugin's reconstruction."""
    rng = np.random.default_rng(1)
    corpus = rng.standard_normal((512, 64)).astype(np.float32)
    q = create_quantizer(PLUGIN, bits=6)
    recon = q.decompress(q.compress(corpus))
    cert = certificate_from_embeddings(corpus, recon, n_anchors=64, seed=0)
    # A 6-bit near-lossless round-trip must yield a non-vacuous, positive floor.
    assert not cert.vacuous, cert.as_dict()
    assert cert.tau_floor > 0.0, cert.as_dict()
