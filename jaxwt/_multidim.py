"""nD wavelet transforms."""
import jax
import jax.numpy as jnp
from jaxwt._dwt import dwt, idwt, dwt_max_level
from jaxwt._filters import get_wavelet


class WaveletCoeffs:
    """Multilevel wavelet decomposition coefficients.

    Custom pytree: approx and details are traced children;
    shapes is static aux_data (not traced/vmapped).
    """

    def __init__(self, approx, details, shapes):
        self.approx = approx
        self.details = details
        self.shapes = shapes

    def __repr__(self):
        return f"WaveletCoeffs(approx={self.approx.shape}, levels={len(self.details)})"


def _wc_flatten(wc):
    children = (wc.approx, wc.details)
    aux = wc.shapes
    return children, aux


def _wc_unflatten(aux, children):
    return WaveletCoeffs(children[0], children[1], aux)


jax.tree_util.register_pytree_node(WaveletCoeffs, _wc_flatten, _wc_unflatten)


def _match_coeff_dims(a, d_dict):
    """Truncate approx to match detail shapes (off by 1 from odd-length signals)."""
    d = next(iter(d_dict.values()))
    return a[tuple(slice(s) for s in d.shape)]


def _dwt_axis(x, wavelet, mode, axis):
    """Apply 1D DWT along one axis of an nD array.

    Uses moveaxis + vmap to map the 1D transform across all other axes.
    This is structural vectorisation (part of the nD algorithm), not batching.
    """
    x = jnp.moveaxis(x, axis, -1)
    shape = x.shape[:-1]
    x_flat = x.reshape(-1, x.shape[-1])
    cA_flat, cD_flat = jax.vmap(lambda row: dwt(row, wavelet, mode))(x_flat)
    cA = jnp.moveaxis(cA_flat.reshape(shape + (-1,)), -1, axis)
    cD = jnp.moveaxis(cD_flat.reshape(shape + (-1,)), -1, axis)
    return cA, cD


def _idwt_axis(cA, cD, wavelet, mode, axis, output_length=None):
    """Apply 1D IDWT along one axis of an nD array."""
    cA = jnp.moveaxis(cA, axis, -1)
    cD = jnp.moveaxis(cD, axis, -1)
    shape = cA.shape[:-1]
    cA_flat = cA.reshape(-1, cA.shape[-1])
    cD_flat = cD.reshape(-1, cD.shape[-1])
    rec_flat = jax.vmap(lambda a, d: idwt(a, d, wavelet, mode, output_length))(cA_flat, cD_flat)
    return jnp.moveaxis(rec_flat.reshape(shape + (-1,)), -1, axis)


def dwtn(data, wavelet, mode='symmetric', axes=None):
    """Single-level nD DWT."""
    if axes is None:
        axes = tuple(range(data.ndim))
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    coeffs = [('', data)]
    for axis in axes:
        new_coeffs = []
        for key, x in coeffs:
            cA, cD = _dwt_axis(x, w, mode, axis)
            new_coeffs.append((key + 'a', cA))
            new_coeffs.append((key + 'd', cD))
        coeffs = new_coeffs
    return dict(sorted(coeffs))


def idwtn(coeffs, wavelet, mode='symmetric', axes=None, target_shape=None):
    """Single-level nD IDWT."""
    keys = sorted(coeffs.keys())
    ndim = len(keys[0])
    if axes is None:
        axes = tuple(range(ndim))
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet

    for i, axis in enumerate(reversed(axes)):
        axis_idx = ndim - 1 - i
        output_length = target_shape[axis] if target_shape is not None else None
        pairs = {}
        for key, arr in coeffs.items():
            base = key[:axis_idx] + key[axis_idx + 1:]
            pairs.setdefault(base, {})[key[axis_idx]] = arr
        coeffs = {
            base: _idwt_axis(pair['a'], pair['d'], w, mode, axis, output_length)
            for base, pair in sorted(pairs.items())
        }
    return coeffs['']


def wavedecn(data, wavelet, mode='symmetric', level=None, axes=None):
    """Multilevel nD DWT."""
    if axes is None:
        axes = tuple(range(data.ndim))
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    if level is None:
        level = dwt_max_level(
            min(data.shape[ax] for ax in axes),
            w.dec_lo.shape[0],
        )
    detail_dicts = []
    shapes = []
    a = data
    for _ in range(level):
        shapes.append(a.shape)
        coeffs = dwtn(a, w, mode, axes)
        a_key = 'a' * len(axes)
        a = coeffs.pop(a_key)
        detail_dicts.append(coeffs)
    detail_dicts.reverse()
    shapes.reverse()
    return WaveletCoeffs(approx=a, details=tuple(detail_dicts), shapes=tuple(shapes))


def waverecn(coeffs, wavelet, mode='symmetric', axes=None):
    """Multilevel nD IDWT."""
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    if axes is None:
        axes = tuple(range(len(next(iter(coeffs.details[0].keys())))))

    a = coeffs.approx
    for detail_dict, target_shape in zip(coeffs.details, coeffs.shapes):
        a = _match_coeff_dims(a, detail_dict)
        a_key = 'a' * len(axes)
        all_coeffs = {a_key: a, **detail_dict}
        a = idwtn(all_coeffs, w, mode, axes, target_shape)
    return a
