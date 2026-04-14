"""Stationary (undecimated) wavelet transform — algorithme à trous."""
import math
import jax
import jax.numpy as jnp
from jaxwt._dwt import dwt, idwt
from jaxwt._filters import get_wavelet, Wavelet


def swt_max_level(input_len):
    """Maximum SWT decomposition level for a given signal length."""
    return int(math.floor(math.log2(input_len)))


def _dilate(filt, dilation):
    """Insert dilation-1 zeros between filter taps."""
    if dilation == 1:
        return filt
    out = jnp.zeros(len(filt) + (len(filt) - 1) * (dilation - 1))
    return out.at[::dilation].set(filt)


def swt(data, wavelet, level=None, start_level=0, axis=-1, trim_approx=False, norm=False):
    """Multilevel 1D stationary wavelet transform.

    Returns [(cA_n, cD_n), ..., (cA_1, cD_1)] or [cA_n, cD_n, ..., cD_1] if trim_approx.
    """
    w = get_wavelet(wavelet)
    if norm:
        w = Wavelet(*(f / jnp.sqrt(2) for f in w))
    if level is None:
        level = swt_max_level(data.shape[axis])

    def _swt_1d(x):
        a = x
        coeffs = []
        for j in range(start_level, start_level + level):
            lo = _dilate(w.dec_lo, 2**j)
            hi = _dilate(w.dec_hi, 2**j)
            F = lo.shape[0]
            xp = jnp.pad(a, (F - 1, F - 1), mode='wrap')
            s = (w.dec_lo.shape[0] // 2) * 2**j
            cA = jnp.convolve(xp, lo, mode='valid')[s:s + a.shape[0]]
            cD = jnp.convolve(xp, hi, mode='valid')[s:s + a.shape[0]]
            coeffs.append((cA, cD))
            a = cA
        coeffs.reverse()
        if trim_approx:
            return [coeffs[0][0]] + [cD for _, cD in coeffs]
        return coeffs

    if data.ndim == 1 or axis == -1 or axis == data.ndim - 1:
        if data.ndim == 1:
            return _swt_1d(data)
        # Apply along last axis via structural vmap
        flat = data.reshape(-1, data.shape[-1])
        results = [_swt_1d(flat[i]) for i in range(flat.shape[0])]
        # Restructure: list of per-level tuples
        if trim_approx:
            return [jnp.stack([r[lev] for r in results]).reshape(data.shape[:-1] + (-1,))
                    for lev in range(len(results[0]))]
        return [(jnp.stack([r[lev][0] for r in results]).reshape(data.shape[:-1] + (-1,)),
                 jnp.stack([r[lev][1] for r in results]).reshape(data.shape[:-1] + (-1,)))
                for lev in range(len(results[0]))]
    # General axis: moveaxis, recurse, moveaxis back
    data = jnp.moveaxis(data, axis, -1)
    result = swt(data, w, level, start_level, -1, trim_approx, False)
    if trim_approx:
        return [jnp.moveaxis(c, -1, axis) for c in result]
    return [(jnp.moveaxis(a, -1, axis), jnp.moveaxis(d, -1, axis)) for a, d in result]


def iswt(coeffs, wavelet, norm=False, axis=-1):
    """Multilevel 1D inverse stationary wavelet transform via cycle spinning."""
    w = get_wavelet(wavelet)
    if norm:
        w = Wavelet(*(f * jnp.sqrt(2) for f in w))

    trim_approx = not isinstance(coeffs[0], (tuple, list))
    output = coeffs[0] if trim_approx else coeffs[0][0]
    if trim_approx:
        detail_coeffs = coeffs[1:]
    else:
        detail_coeffs = coeffs

    num_levels = len(detail_coeffs)

    for j in range(num_levels, 0, -1):
        step_size = 2**(j - 1)
        cD = detail_coeffs[-j] if trim_approx else detail_coeffs[-j][1]
        N = output.shape[0]

        for first in range(step_size):
            indices = jnp.arange(first, N, step_size)
            even_idx = indices[0::2]
            odd_idx = indices[1::2]

            x1 = idwt(output[even_idx], cD[even_idx], w, 'periodization')
            x2 = jnp.roll(idwt(output[odd_idx], cD[odd_idx], w, 'periodization'), 1)
            output = output.at[indices].set((x1 + x2) / 2)

    return output
