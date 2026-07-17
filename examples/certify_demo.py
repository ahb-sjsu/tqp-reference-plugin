# tqp-reference-plugin: certification integration demo
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Compress a corpus with the out-of-tree plugin, then certify it two ways.

Shows the Phase-5 "certification integration" deliverable: the plugin's
reconstruction is certified by `certificate_from_embeddings` directly, and by
the `tqp certify` CLI on saved arrays.

    python examples/certify_demo.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from turboquant_pro import certificate_from_embeddings, create_quantizer


def main() -> int:
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((2000, 100)).astype(np.float32)

    q = create_quantizer("tqp_reference", bits=6)
    recon = q.decompress(q.compress(corpus))

    # (1) In-process: the same instrument the CLI wraps.
    cert = certificate_from_embeddings(corpus, recon, n_anchors=128, seed=0)
    print("in-process certificate:")
    print(
        f"  kappa={cert.kappa:.4f}  tau_floor={cert.tau_floor:.4f}  "
        f"vacuous={cert.vacuous}"
    )

    # (2) Via the `tqp certify` CLI on saved arrays.
    with tempfile.TemporaryDirectory() as d:
        o, r = Path(d) / "orig.npy", Path(d) / "recon.npy"
        np.save(o, corpus)
        np.save(r, recon)
        print("\ntqp certify:")
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "turboquant_pro.cli",
                "certify",
                "--original",
                str(o),
                "--reconstructed",
                str(r),
                "--min-tau",
                "0.5",
                "--format",
                "text",
            ],
            check=False,
        )
    return rc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
