"""Core 1D discrete wavelet transform."""
import math
import jax.numpy as jnp
from jaxwt._filters import get_wavelet


def dwt_max_level(data_len, filter_len):
    """floor(log2(data_len / (filter_len - 1)))"""
    return int(math.floor(math.log2(data_len / (filter_len - 1))))


def dwt(x, wavelet, mode='symmetric'):
    """1D discrete wavelet transform."""
    w = get_wavelet(wavelet)
    F = w.dec_lo.shape[0]
    xp = jnp.pad(x, (F - 2, F - 1), mode=mode)
    return jnp.convolve(xp, w.dec_lo, mode='valid')[::2], jnp.convolve(xp, w.dec_hi, mode='valid')[::2]


def idwt(cA, cD, wavelet, mode='symmetric'):
    """1D inverse discrete wavelet transform."""
    w = get_wavelet(wavelet)
    return _upc(cA, w.rec_lo) + _upc(cD, w.rec_hi)


def _upc(c, f):
    """Upsample-convolve: even/odd filter splitting."""
    e = jnp.convolve(c, f[::2], mode='valid')
    o = jnp.convolve(c, f[1::2], mode='valid')
    return jnp.empty(2 * e.shape[0]).at[::2].set(e).at[1::2].set(o)
