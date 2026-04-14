"""nD wavelet transforms."""
import jax
import jax.numpy as jnp
from jaxwt._dwt import dwt, idwt, dwt_max_level
from jaxwt._filters import get_wavelet


class WaveletCoeffs:
    """Multilevel wavelet coefficients. approx and details are traced; shapes and axes are static."""
    __slots__ = ('approx', 'details', 'shapes', 'axes')

    def __init__(self, approx, details, shapes, axes):
        self.approx = approx
        self.details = details
        self.shapes = shapes
        self.axes = tuple(axes)


jax.tree_util.register_pytree_node(
    WaveletCoeffs,
    lambda wc: ((wc.approx, wc.details), (wc.shapes, wc.axes)),
    lambda aux, children: WaveletCoeffs(children[0], children[1], aux[0], aux[1]),
)


def _dwt_axis(x, w, mode, axis):
    """1D DWT along one axis via structural vmap."""
    x = jnp.moveaxis(x, axis, -1)
    flat = x.reshape(-1, x.shape[-1])
    cA, cD = jax.vmap(lambda row: dwt(row, w, mode))(flat)
    return (jnp.moveaxis(cA.reshape(x.shape[:-1] + (-1,)), -1, axis),
            jnp.moveaxis(cD.reshape(x.shape[:-1] + (-1,)), -1, axis))


def _idwt_axis(cA, cD, w, mode, axis):
    """1D IDWT along one axis via structural vmap."""
    cA, cD = jnp.moveaxis(cA, axis, -1), jnp.moveaxis(cD, axis, -1)
    rec = jax.vmap(lambda a, d: idwt(a, d, w, mode))(
        cA.reshape(-1, cA.shape[-1]), cD.reshape(-1, cD.shape[-1]))
    return jnp.moveaxis(rec.reshape(cA.shape[:-1] + (-1,)), -1, axis)


def dwtn(data, wavelet, mode='symmetric', axes=None):
    """Single-level nD DWT."""
    axes = tuple(range(data.ndim)) if axes is None else tuple(axes)
    w = get_wavelet(wavelet)
    coeffs = [('', data)]
    for axis in axes:
        coeffs = [(k + s, c) for k, x in coeffs
                  for s, c in zip('ad', _dwt_axis(x, w, mode, axis))]
    return dict(sorted(coeffs))


def idwtn(coeffs, wavelet, mode='symmetric', axes=None):
    """Single-level nD IDWT. Reverses dwtn by recombining 'a'/'d' pairs along each axis."""
    axes = tuple(range(len(next(iter(coeffs))))) if axes is None else tuple(axes)
    w = get_wavelet(wavelet)
    for axis_pos, axis in zip(reversed(range(len(axes))), reversed(axes)):
        grouped = {}
        for key, arr in coeffs.items():
            rest = key[:axis_pos] + key[axis_pos + 1:]
            grouped.setdefault(rest, {})[key[axis_pos]] = arr
        coeffs = {rest: _idwt_axis(ad['a'], ad['d'], w, mode, axis)
                  for rest, ad in sorted(grouped.items())}
    return coeffs['']


def wavedecn(data, wavelet, mode='symmetric', level=None, axes=None):
    """Multilevel nD DWT."""
    axes = tuple(range(data.ndim)) if axes is None else tuple(axes)
    w = get_wavelet(wavelet)
    if level is None:
        level = dwt_max_level(min(data.shape[ax] for ax in axes), w.dec_lo.shape[0])
    details, shapes = [], []
    a = data
    for _ in range(level):
        shapes.append(a.shape)
        c = dwtn(a, w, mode, axes)
        a = c.pop('a' * len(axes))
        details.append(c)
    details.reverse()
    shapes.reverse()
    return WaveletCoeffs(a, tuple(details), tuple(shapes), axes)


def waverecn(coeffs, wavelet, mode='symmetric'):
    """Multilevel nD IDWT."""
    w = get_wavelet(wavelet)
    axes, a_key = coeffs.axes, 'a' * len(coeffs.axes)
    a = coeffs.approx
    for d, shape in zip(coeffs.details, coeffs.shapes):
        d_shape = next(iter(d.values())).shape
        a = idwtn({a_key: a[tuple(slice(s) for s in d_shape)], **d}, w, mode, axes)
        a = a[tuple(slice(s) for s in shape)]
    return a


# --- 2D convenience wrappers ---

def dwt2(data, wavelet, mode='symmetric', axes=(-2, -1)):
    """2D DWT. Returns (cA, (cH, cV, cD))."""
    c = dwtn(data, wavelet, mode, axes)
    return c['aa'], (c['da'], c['ad'], c['dd'])


def idwt2(coeffs, wavelet, mode='symmetric', axes=(-2, -1)):
    """2D IDWT from (cA, (cH, cV, cD)) format."""
    cA, (cH, cV, cD) = coeffs
    return idwtn({'aa': cA, 'da': cH, 'ad': cV, 'dd': cD}, wavelet, mode, axes)


def wavedec2(data, wavelet, mode='symmetric', level=None, axes=(-2, -1)):
    """Multilevel 2D DWT. Returns [cA, (cH,cV,cD)_n, ..., (cH,cV,cD)_1]."""
    axes = tuple(axes)
    w = get_wavelet(wavelet)
    if level is None:
        level = dwt_max_level(min(data.shape[ax] for ax in axes), w.dec_lo.shape[0])
    result = []
    a = data
    for _ in range(level):
        a, ds = dwt2(a, w, mode, axes)
        result.append(ds)
    result.append(a)
    result.reverse()
    return result


def waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1)):
    """Multilevel 2D IDWT from [cA, (cH,cV,cD)_n, ..., (cH,cV,cD)_1]."""
    w = get_wavelet(wavelet)
    a = coeffs[0]
    for cH, cV, cD in coeffs[1:]:
        # Trim approx to detail shape
        d_shape = cH.shape
        a = a[tuple(slice(s) for s in d_shape)]
        a = idwt2((a, (cH, cV, cD)), w, mode, axes)
    return a
