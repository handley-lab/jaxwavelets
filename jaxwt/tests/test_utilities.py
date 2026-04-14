"""Tests for filter utilities and partial DWT functions."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pywt
import pytest

import jaxwt

WAVELETS = ['haar', 'db2', 'db4', 'db8', 'sym4', 'coif2']
ATOL = 1e-11


# --- qmf ---

@pytest.mark.parametrize('wavelet', WAVELETS)
def test_qmf(wavelet):
    w = pywt.Wavelet(wavelet)
    result = jaxwt.qmf(jnp.array(w.rec_lo))
    expected = pywt.qmf(w.rec_lo)
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)


# --- orthogonal_filter_bank ---

@pytest.mark.parametrize('wavelet', WAVELETS)
def test_orthogonal_filter_bank(wavelet):
    w = pywt.Wavelet(wavelet)
    bank_jax = jaxwt.orthogonal_filter_bank(jnp.array(w.rec_lo))
    bank_pywt = pywt.orthogonal_filter_bank(w.rec_lo)
    for j, p in zip(bank_jax, bank_pywt):
        np.testing.assert_allclose(np.array(j), p, atol=ATOL)


# --- downcoef ---

@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [8, 16, 32])
@pytest.mark.parametrize('part', ['a', 'd'])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_downcoef(wavelet, N, part, level):
    x_np = np.random.RandomState(0).randn(N)
    result = jaxwt.downcoef(part, jnp.array(x_np), wavelet, level=level)
    expected = pywt.downcoef(part, x_np, wavelet, level=level)
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)


# --- upcoef ---

@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('part', ['a', 'd'])
@pytest.mark.parametrize('level', [1, 2])
def test_upcoef(wavelet, part, level):
    x_np = np.random.RandomState(0).randn(16)
    cA, cD = pywt.dwt(x_np, wavelet)
    coeffs = cA if part == 'a' else cD
    result = jaxwt.upcoef(part, jnp.array(coeffs), wavelet, level=level)
    expected = pywt.upcoef(part, coeffs, wavelet, level=level)
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)


@pytest.mark.parametrize('wavelet', WAVELETS)
def test_upcoef_take(wavelet):
    x_np = np.random.RandomState(0).randn(16)
    cA, cD = pywt.dwt(x_np, wavelet)
    result = jaxwt.upcoef('a', jnp.array(cA), wavelet, take=len(x_np))
    expected = pywt.upcoef('a', cA, wavelet, take=len(x_np))
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)
