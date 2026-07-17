# tqp-reference-plugin

[![Tests](https://img.shields.io/github/actions/workflow/status/ahb-sjsu/tqp-reference-plugin/ci.yml?label=tests)](https://github.com/ahb-sjsu/tqp-reference-plugin/actions)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/ahb-sjsu/tqp-reference-plugin)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![turboquant-pro](https://img.shields.io/badge/turboquant--pro-%E2%89%A51.8-8A2BE2.svg)](https://github.com/ahb-sjsu/turboquant-pro)

The **reference out-of-tree quantizer plugin** for
[turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro).

It exists for one job: to prove — from a package that lives *outside* the main
repository — that the turboquant-pro plugin contract actually works end to end.
A quantizer here registers through the `turboquant_pro.plugins` entry point,
passes the executable conformance kit, and is certified by the same rank
certificate the rest of the toolkit uses. **No code in the main tree.**

It is deliberately not novel. If you want a real production format, see the
in-tree incubators (`tqp-bnb`, `tqp-gptq-awq`, `tqp-trtllm`). This one is the
worked example the docs point at, and the smallest thing that can pass every
applicable check.

## Install

Needs **turboquant-pro ≥ 1.8** — the release that ships the plugin protocol, the
conformance kit, and the `tqp` CLI. Until 1.8 is on PyPI, install turboquant-pro
from git:

```bash
pip install "turboquant-pro @ git+https://github.com/ahb-sjsu/turboquant-pro.git@master"
pip install .            # from a checkout of this repo
# once 1.8 is released:  pip install turboquant-pro tqp-reference-plugin
```

## It just shows up

Installing the package is all it takes — turboquant-pro discovers it via the
entry point:

```bash
$ tqp plugin list
NAME           TIER          TARGETS
per_channel    beta          kv_key
polar          stable        kv_value
tqp_reference  experimental  embedding, kv_key, weight   # <- from this package

$ tqp plugin conformance tqp_reference
== tqp_reference ==
conformance PASSED
  roundtrip: pass (rel err 0.0193)
  packed: pass
  affine: pass (max |diff| 4.77e-07 <= 5.72e-06)
  csr: skip: no sparse overlay
  serialization: pass
```

## Capability table

| Capability | Status | Notes |
|---|---|---|
| Targets | `embedding`, `kv_key`, `weight` | per-channel symmetric uniform |
| `compress` / `decompress` | ✅ | any `(…, D)` array; KV block `(1,H,S,D)` |
| Affine contract (`grid_params`/`codes`) | ✅ exact | `mu=0`, `weight`=per-channel scale → **fused-decode eligible** |
| Bit-packing (`packed=True`) | ✅ honest | codes stored `bits` wide, container self-reports `packed` |
| Serialization (`to_bytes`/`from_bytes`) | ✅ | NumPy-`savez` container round-trip |
| Sparse overlay (`outlier_csr`) | — | none; CSR check skips (see in-tree LLM.int8) |
| Native dtype passthrough | — | `None` |
| Hardware assumptions | none | pure NumPy, CPU |
| Maturity | experimental | reference/proof, not a production recipe |

Because the affine contract is exact, `tqp_reference` inherits the M4 fused
compute-on-codes decode with zero kernel work — the same property the in-tree
`per_channel` format has, demonstrated here from outside the tree.

## Write your own plugin in 20 minutes

A turboquant-pro quantizer is any object with `compress` and `decompress`.
Everything else is optional and probed, never required.

**1. The quantizer** (`my_plugin/plugin.py`):

```python
import numpy as np
from turboquant_pro.plugins import PluginSpec, TARGET_EMBEDDING

class MyQuantizer:
    def __init__(self, bits: int = 8, **_ignored):
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1

    def compress(self, x, packed=False):        # packed is optional
        x = np.asarray(x, np.float32)
        scale = np.maximum(np.abs(x).max(0), 1e-12) / self.qmax
        codes = np.clip(np.rint(x / scale), -self.qmax, self.qmax).astype(np.int8)
        return {"codes": codes, "scale": scale.astype(np.float32)}

    def decompress(self, c):
        return (c["codes"] * c["scale"]).astype(np.float32)

SPEC = PluginSpec(
    name="my_quantizer",
    factory=MyQuantizer,
    targets=frozenset({TARGET_EMBEDDING}),
    tier="experimental",
    description="my first turboquant-pro plugin",
)
```

**2. Register it** (`pyproject.toml`):

```toml
[project.entry-points."turboquant_pro.plugins"]
my_quantizer = "my_plugin.plugin:SPEC"
```

**3. Prove it** — `pip install .`, then:

```python
import numpy as np
from turboquant_pro import available_plugins, create_quantizer
from turboquant_pro.plugin_conformance import assert_conformance

assert "my_quantizer" in available_plugins()          # entry-point discovery
q = create_quantizer("my_quantizer", bits=8)
assert_conformance(q, np.random.randn(256, 64).astype("f4"))   # container contract
```

That is the whole loop: implement two methods, add one entry-point line, and the
entire certification + conformance toolkit works on your format on day one. Add
`grid_params`/`codes` when you want the fused decode (see `plugin.py` in this
repo for the exact, conformance-passing form).

For the deep version — every optional capability, all five conformance checks
explained, certification, testing, and publishing — see
**[`docs/GUIDE.md`](docs/GUIDE.md)**.

## Certification

The plugin's output is certifiable by the same instrument `tqp certify` uses:

```python
from turboquant_pro import certificate_from_embeddings, create_quantizer
q = create_quantizer("tqp_reference", bits=6)
recon = q.decompress(q.compress(corpus))
cert = certificate_from_embeddings(corpus, recon)      # rank certificate
print(cert.tau_floor, cert.vacuous)
```

Or via the CLI on saved arrays — see [`examples/certify_demo.py`](examples/certify_demo.py).

## Development

```bash
pip install turboquant-pro
pip install -e ".[dev]"
pytest -q
ruff check . && black --check .
```

MIT licensed. Part of the turboquant-pro ecosystem.
