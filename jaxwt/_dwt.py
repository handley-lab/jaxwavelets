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
    if mode == 'periodization':
        if x.shape[0] % 2:
            x = jnp.concatenate([x, x[-1:]])
        xp = jnp.pad(x, (F // 2 - 1, F // 2 - 1), mode='wrap')
        M = int(math.ceil(x.shape[0] / 2))
        return jnp.convolve(xp, w.dec_lo, mode='valid')[::2][:M], jnp.convolve(xp, w.dec_hi, mode='valid')[::2][:M]
    xp = jnp.pad(x, (F - 2, F - 1), mode=mode)
    return jnp.convolve(xp, w.dec_lo, mode='valid')[::2], jnp.convolve(xp, w.dec_hi, mode='valid')[::2]


def idwt(cA, cD, wavelet, mode='symmetric'):
    """1D inverse discrete wavelet transform."""
    w = get_wavelet(wavelet)
    if mode == 'periodization':
        return _upc_per(cA, w.rec_lo) + _upc_per(cD, w.rec_hi)
    return _upc(cA, w.rec_lo) + _upc(cD, w.rec_hi)


def downcoef(part, data, wavelet, mode='symmetric', level=1):
    """Partial DWT: extract single subband at the given decomposition level.

    'a': low-pass level times. 'd': low-pass level-1 times, then high-pass once.
    """
    w = get_wavelet(wavelet)
    for _ in range(level - 1):
        data, _ = dwt(data, w, mode)
    cA, cD = dwt(data, w, mode)
    return cA if part == 'a' else cD


def upcoef(part, coeffs, wavelet, level=1, take=0):
    """Partial IDWT: reconstruct from single subband ('a' or 'd') for level levels.

    part='a': rec_lo at every level. part='d': rec_hi first, then rec_lo.
    """
    w = get_wavelet(wavelet)
    rec = _upcoef_step(coeffs, w.rec_lo if part == 'a' else w.rec_hi)
    for _ in range(level - 1):
        rec = _upcoef_step(rec, w.rec_lo)
    if take > 0:
        rec = rec[len(rec) // 2 - take // 2:len(rec) // 2 + take // 2 + take % 2]
    return rec


def _upcoef_step(c, f):
    """Single upsample-by-2 then full convolve."""
    return jnp.convolve(jnp.zeros(2 * c.shape[0] - 1).at[::2].set(c), f)


def _upc(c, f):
    """Upsample-convolve: even/odd filter splitting."""
    e = jnp.convolve(c, f[::2], mode='valid')
    o = jnp.convolve(c, f[1::2], mode='valid')
    return jnp.stack([e, o], axis=1).reshape(-1)


def _upc_per(c, f):
    """Upsample-convolve for periodization: circular convolution."""
    F, L = f.shape[0], 2 * c.shape[0]
    u = jnp.zeros(L).at[::2].set(c)
    z = jnp.convolve(jnp.pad(u, (F - 1, F - 1), mode='wrap'), f, mode='valid')
    return jnp.roll(z[:L], -(F // 2 - 1))
