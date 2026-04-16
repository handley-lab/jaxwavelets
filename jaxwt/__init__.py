"""jaxwt: JAX-native wavelet transforms."""

from jaxwt._cwt import (
    ContinuousWavelet,
    CWTKernelBank,
    apply_cwt,
    central_frequency,
    cwt,
    integrate_wavelet,
    prepare_cwt,
    scale2frequency,
    wavefun,
)
from jaxwt._dwt import downcoef, dwt, dwt_max_level, idwt, upcoef
from jaxwt._filters import Wavelet, get_wavelet, orthogonal_filter_bank, qmf
from jaxwt._fswt import FswavedecnResult, fswavedecn, fswaverecn
from jaxwt._mra import imra, imra2, imran, mra, mra2, mran
from jaxwt._multidim import (
    WaveletCoeffs,
    dwt2,
    dwtn,
    idwt2,
    idwtn,
    wavedec2,
    wavedecn,
    waverec2,
    waverecn,
)
from jaxwt._packets import (
    wp_decompose,
    wp_decompose_nd,
    wp_reconstruct,
    wp_reconstruct_nd,
)
from jaxwt._swt import iswt, iswt2, iswtn, swt, swt2, swt_max_level, swtn
from jaxwt._thresholding import threshold, threshold_firm
