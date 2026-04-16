# jaxwt

JAX-native wavelet transforms. Differentiable, JIT-compilable, GPU-ready.

## Features

Full feature parity with [PyWavelets](https://pywavelets.readthedocs.io/):

| Transform | Functions |
|---|---|
| Discrete wavelet | `dwt`, `idwt`, `dwt2`, `idwt2`, `dwtn`, `idwtn` |
| Multilevel | `wavedec2`, `waverec2`, `wavedecn`, `waverecn` |
| Stationary (undecimated) | `swt`, `iswt`, `swt2`, `iswt2`, `swtn`, `iswtn` |
| Continuous | `cwt`, `prepare_cwt`, `apply_cwt` |
| Fully separable | `fswavedecn`, `fswaverecn` |
| Multiresolution analysis | `mra`, `imra`, `mra2`, `imra2`, `mran`, `imran` |
| Wavelet packets | `wp_decompose`, `wp_reconstruct`, `wp_decompose_nd`, `wp_reconstruct_nd` |
| Thresholding | `threshold`, `threshold_firm` |
| Utilities | `downcoef`, `upcoef`, `qmf`, `orthogonal_filter_bank` |

**Wavelets:** haar, db1-20, sym2-20, coif1-5, plus continuous wavelets (Morlet, Mexican hat, Gaussian 1-8, complex Gaussian 1-8, complex Morlet, Shannon, frequency B-spline).

## Usage

```python
import jax
import jax.numpy as jnp
import jaxwt

# Single example
x = jnp.ones((64, 64))
coeffs = jaxwt.wavedecn(x, 'db4', level=3)
rec = jaxwt.waverecn(coeffs, 'db4')

# Batched via vmap
from functools import partial
batch = jnp.ones((10, 64, 64))
batch_coeffs = jax.vmap(partial(jaxwt.wavedecn, wavelet='db4', level=3))(batch)

# Differentiable
grad = jax.grad(lambda x: jnp.sum(jaxwt.waverecn(jaxwt.wavedecn(x, 'db4'), 'db4')))(x)

# JIT-compiled
fast = jax.jit(jaxwt.wavedecn, static_argnames=['wavelet', 'mode', 'level'])
coeffs = fast(x, wavelet='db4', level=3)
```

## Performance

JIT-compiled jaxwt on CPU vs pywt C (lower is better):

```
Transform                       pywt         jaxwt (JIT)    ratio
--------------------------------------------------------------------------
dwt 1D (N=4096)                  0.011ms       0.023ms       2.1x
wavedecn 1D (N=4096)             0.065ms       0.046ms       0.7x  ← faster
dwt2 (256x256)                   0.608ms       0.287ms       0.5x  ← faster
wavedecn 2D level=3              0.755ms       0.363ms       0.5x  ← faster
swt 1D level=3 (N=1024)          0.023ms       0.025ms       1.1x
cwt morl 6 scales (N=512)        0.316ms       0.139ms       0.4x  ← faster
cwt cmor 6 scales (N=512)        0.615ms       0.254ms       0.4x  ← faster
```

Plus: `jax.grad`, `jax.vmap`, `jax.pmap`, GPU acceleration — none of which pywt supports.

## Installation

```bash
pip install jax jaxlib
# Then add jaxwt to your path or install from source
```

No runtime dependency on pywt. Filter coefficients are pre-extracted.

## Testing

```bash
pip install pywt pytest
pytest jaxwt/tests/
```

1182 tests verify numerical agreement with pywt to machine precision (`atol=1e-14` for direct comparisons, `atol=1e-11` for roundtrips).

## Design

- **Pure JAX** — no numpy, no C extensions, no complex arithmetic internally
- **Single-example functions** — compose with `jax.vmap`/`jax.pmap`/`jax.grad`/`jax.jit` from outside
- **NamedTuple pytrees** — coefficients are JAX-compatible pytrees
- **No defensive programming** — clean, lightweight scientific code
