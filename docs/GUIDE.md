# Writing a turboquant-pro quantizer plugin

A complete guide to the plugin contract, using `tqp-reference-plugin` as the
worked example. If you just want the fastest path, the README's
["Write your own plugin in 20 minutes"](../README.md#write-your-own-plugin-in-20-minutes)
is enough; this is the deep version.

## 1. What a plugin is

turboquant-pro is not just a quantizer — it is a *certification and fused-decode
substrate* that production quantization recipes plug into. A plugin is any
object that can `compress` a float array to an opaque container and `decompress`
it back. That is the entire required surface:

```python
class Quantizer(Protocol):
    def compress(self, x: np.ndarray, **kwargs) -> Any: ...
    def decompress(self, c: Any) -> np.ndarray: ...
```

Because turboquant-pro's instruments (the rank certificate, the `a2_probe`
consumer probe, `behavioral_agreement`) are *consumer-metric* tools — they only
need round-trips, not internals — a quantizer that satisfies this protocol is
certifiable on day one. Everything else is **optional and probed**, never
required.

## 2. Registering out of tree

A plugin lives in its own package and registers through the
`turboquant_pro.plugins` entry-point group. turboquant-pro enumerates that group
and registers whatever `PluginSpec` it finds — with no code in the main tree:

```toml
# pyproject.toml
[project.entry-points."turboquant_pro.plugins"]
tqp_reference = "tqp_reference.plugin:SPEC"
```

```python
# tqp_reference/plugin.py
from turboquant_pro.plugins import PluginSpec, TARGET_EMBEDDING

SPEC = PluginSpec(
    name="tqp_reference",              # registry key; no "/"
    factory=MyQuantizer,               # factory(**config) -> Quantizer
    targets=frozenset({TARGET_EMBEDDING}),
    tier="experimental",               # out-of-tree plugins enter here
    description="...",
)
```

After `pip install`, it appears in `tqp plugin list` and is instantiable with
`create_quantizer("tqp_reference", **config)` — discovered purely through the
entry point, with no import of your package by the caller.

### Targets

Declare which tensors your format is for, as plain strings (never a moving
enum): `TARGET_EMBEDDING`, `TARGET_KV_KEY`, `TARGET_KV_VALUE`, `TARGET_WEIGHT`.
Keys and values are split deliberately — they have opposite disciplines (the
v1.2.0 keys finding), so a values format should not claim `kv_key`.

## 3. The optional capabilities

Each is discovered with a module-level probe; returning `None` (or omitting the
method) is always a safe, correct degrade — never an error.

### Affine capability — the fused-decode gate

If your container can be written as `dequant = mu + weight * grid[code]`
(optionally plus a sparse overlay), expose it:

```python
def grid_params(self, c):   # -> (mu (H,D), weight (H,D), grid (L,)) or None
def codes(self, c):         # -> (B, H, S, D) uint8 grid indices
def outlier_csr(self, c):   # -> (row_ptr, cols, deltas) or None
```

under the KV block convention `(B=1, H, S, D)`. A format that exposes these
**exactly** inherits the M4 fused compute-on-codes decode with zero kernel work,
because keys enter attention only through `q·k`, so score-level equality follows
algebraically from container-level equality. This is the single highest-value
capability to implement; `tqp-reference-plugin` does it with `mu = 0` and
`weight` the per-channel scale.

### Native dtype passthrough

```python
def native_dtype(self):     # -> "float8_e4m3fn" | None
```

Name a hardware dtype for passthrough execution where the silicon fuses the
dequant itself. Return `None` if not applicable.

### Packing and serialization

- `compress(x, packed=True)` — return a bit-packed container that
  `decompress`es identically. Have the container self-report `packed` (a `bool`
  attribute) so conformance can confirm the packed path was actually exercised.
- `container.to_bytes()` / `Container.from_bytes(b)` — a byte round-trip that
  decompresses identically. Lets your format ride the TQE envelope later.

## 4. Conformance — the contract is executable

The contract is not prose; it is `turboquant_pro.plugin_conformance`. Run it in
your CI:

```python
from turboquant_pro.plugin_conformance import assert_conformance
assert_conformance(my_quantizer, x)     # x: (1, H, S, D) float32
```

or over the CLI, exactly as a user would:

```bash
tqp plugin conformance my_quantizer
```

Each check returns pass / skip-with-reason / FAIL-with-detail:

| Check | What it verifies |
|---|---|
| `roundtrip` | `decompress(compress(x))` has the right shape, is finite, and sits inside a coarse relative-error envelope. |
| `packed` | if `compress(packed=True)` is accepted, packed and unpacked decompress identically — and the packed path was genuinely exercised. |
| `affine` | if `grid_params` + `codes` are exposed, `mu + weight*grid[codes]` (+ overlay) reproduces `decompress` **exactly** (scale-aware tolerance). This is the fused-decode safety gate. |
| `csr` | if `outlier_csr` is non-`None`, the sparse overlay is structurally valid (monotone `row_ptr`, in-range columns). |
| `serialization` | if the container exposes `to_bytes`/`from_bytes`, a byte round-trip decompresses identically. |

`tqp-reference-plugin` passes roundtrip, packed, affine, and serialization; csr
skips (it has no sparse overlay — see the in-tree `bnb_llm_int8` format for a CSR
example).

## 5. Certification

The instruments certify your output the same way for every plugin:

```python
from turboquant_pro import certificate_from_embeddings, create_quantizer
q = create_quantizer("my_quantizer", bits=6)
recon = q.decompress(q.compress(corpus))
cert = certificate_from_embeddings(corpus, recon)   # distribution-free rank certificate
print(cert.tau_floor, cert.spearman_floor, cert.vacuous)
```

`cert.tau_floor` is a **guaranteed** Kendall-τ floor on how well the compressed
ranking tracks the exact one — the metric retrieval actually consumes. Acceptance
is this floor (or recall@k), **never reconstruction cosine**, which this project
shows can read ~0.97 while the ranking collapses. See
[`examples/certify_demo.py`](../examples/certify_demo.py) for the same thing via
the `tqp certify` CLI on saved arrays.

## 6. Testing checklist

Mirror `tests/` in this repo:

- **conformance** across the bit widths / configs you support (`assert_conformance`);
- **discovery** — assert your plugin is in `available_plugins()` *without importing
  your own module*, proving the entry point works;
- **certification** — a near-lossless config yields a non-vacuous, positive `tau_floor`;
- **semantics** — any exact properties (e.g. the affine reconstruction equals
  `decompress`), and that invalid inputs raise cleanly rather than crash.

Wire `pytest -q`, `ruff check .`, and `black --check .` into CI that installs
turboquant-pro and runs `tqp plugin conformance <name>`.

## 7. Maturity and promotion

Out-of-tree plugins enter at `tier="experimental"`. They promote to `beta` /
`stable` per `docs/api-stability.md` in the main repo, on conformance plus one
public-data validation. Declare a capability table in your README (target
tensors, affine support, hardware assumptions, maturity) so a user can see at a
glance what your format guarantees.

## 8. Publishing

1. `pip install build && python -m build` → a wheel + sdist.
2. Until turboquant-pro 1.8 is on PyPI, depend on `turboquant-pro>=1.8` and have
   consumers install it from git first (see the README install section); the
   plugin then installs cleanly with its entry point registered.
3. Publish to PyPI (`twine upload`) or share the wheel; `pip install your-plugin`
   is all a user needs — turboquant-pro discovers it automatically.

---

For the authoritative protocol, read `turboquant_pro/plugins.py` and
`turboquant_pro/plugin_conformance.py` in the main repo, and
`docs/DESIGN_hardware_and_plugins.md` for the design rationale.
