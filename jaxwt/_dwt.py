"""Core 1D discrete wavelet transform."""
import math
import jax.numpy as jnp
from jaxwt._filters import get_wavelet

_PAD_MODES = {
    'symmetric': 'symmetric',
    'reflect': 'reflect',
}


def dwt_max_level(data_len, filter_len):
    """Maximum useful decomposition level: floor(log2(data_len / (filter_len - 1))).

    Pure Python — intended for static use, not inside jit-traced code.
    """
    return int(math.floor(math.log2(data_len / (filter_len - 1))))


def dwt(x, wavelet, mode='symmetric'):
    """1D discrete wavelet transform."""
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    F = w.dec_lo.shape[0]
    x_padded = jnp.pad(x, (F - 2, F - 1), mode=_PAD_MODES[mode])
    cA = jnp.convolve(x_padded, w.dec_lo, mode='valid')[::2]
    cD = jnp.convolve(x_padded, w.dec_hi, mode='valid')[::2]
    return cA, cD


def idwt(cA, cD, wavelet, mode='symmetric', output_length=None):
    """1D inverse discrete wavelet transform.

    Parameters
    ----------
    cA, cD : approximation and detail coefficients
    wavelet : str or Wavelet NamedTuple
    mode : str
    output_length : int, optional
        Expected output length. If given, the result is trimmed to this length.
    """
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    rec = _upsample_convolve(cA, w.rec_lo) + _upsample_convolve(cD, w.rec_hi)
    if output_length is not None:
        rec = rec[:output_length]
    return rec


def _upsample_convolve(coeffs, rec_filter):
    """Upsample by 2 and convolve via even/odd filter splitting."""
    filt_even, filt_odd = rec_filter[::2], rec_filter[1::2]
    out_even = jnp.convolve(coeffs, filt_even, mode='valid')
    out_odd = jnp.convolve(coeffs, filt_odd, mode='valid')
    result = jnp.empty(2 * out_even.shape[0])
    return result.at[::2].set(out_even).at[1::2].set(out_odd)
