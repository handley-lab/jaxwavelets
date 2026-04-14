"""Tests for stationary wavelet transform."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pywt
import pytest

from jaxwt._swt import swt, iswt

WAVELETS = ['haar', 'db2', 'db4', 'sym4']
ATOL = 1e-11


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [8, 16, 32, 64])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_swt_matches_pywt(wavelet, N, level):
    if 2**level > N:
        pytest.skip("level too high for signal length")
    x_np = np.random.RandomState(0).randn(N)
    coeffs_jax = swt(jnp.array(x_np), wavelet, level=level)
    coeffs_pywt = pywt.swt(x_np, wavelet, level=level)
    for (cA_j, cD_j), (cA_p, cD_p) in zip(coeffs_jax, coeffs_pywt):
        np.testing.assert_allclose(np.array(cA_j), cA_p, atol=ATOL)
        np.testing.assert_allclose(np.array(cD_j), cD_p, atol=ATOL)


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [8, 16, 32])
def test_iswt_roundtrip(wavelet, N):
    x = jnp.array(np.random.RandomState(0).randn(N))
    level = min(3, int(np.log2(N)))
    coeffs = swt(x, wavelet, level=level)
    rec = iswt(coeffs, wavelet)
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL)


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [8, 16, 32])
def test_iswt_matches_pywt(wavelet, N):
    x_np = np.random.RandomState(0).randn(N)
    level = min(3, int(np.log2(N)))
    coeffs_pywt = pywt.swt(x_np, wavelet, level=level)
    rec_pywt = pywt.iswt(coeffs_pywt, wavelet)
    rec_jax = iswt([(jnp.array(a), jnp.array(d)) for a, d in coeffs_pywt], wavelet)
    np.testing.assert_allclose(np.array(rec_jax), rec_pywt, atol=ATOL)


def test_swt_trim_approx(wavelet='db2', N=16, level=2):
    x_np = np.random.RandomState(0).randn(N)
    coeffs_jax = swt(jnp.array(x_np), wavelet, level=level, trim_approx=True)
    coeffs_pywt = pywt.swt(x_np, wavelet, level=level, trim_approx=True)
    for j, p in zip(coeffs_jax, coeffs_pywt):
        np.testing.assert_allclose(np.array(j), p, atol=ATOL)
