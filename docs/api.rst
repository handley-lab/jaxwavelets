API Reference
=============

Discrete Wavelet Transform
--------------------------

.. autofunction:: jaxwt.dwt
.. autofunction:: jaxwt.idwt
.. autofunction:: jaxwt.dwt2
.. autofunction:: jaxwt.idwt2
.. autofunction:: jaxwt.dwtn
.. autofunction:: jaxwt.idwtn

Multilevel DWT
--------------

.. autofunction:: jaxwt.wavedec2
.. autofunction:: jaxwt.waverec2
.. autofunction:: jaxwt.wavedecn
.. autofunction:: jaxwt.waverecn

.. autoclass:: jaxwt.WaveletCoeffs
   :members:

Stationary Wavelet Transform
-----------------------------

.. autofunction:: jaxwt.swt
.. autofunction:: jaxwt.iswt
.. autofunction:: jaxwt.swt2
.. autofunction:: jaxwt.iswt2
.. autofunction:: jaxwt.swtn
.. autofunction:: jaxwt.iswtn

Continuous Wavelet Transform
-----------------------------

.. autofunction:: jaxwt.cwt
.. autofunction:: jaxwt.prepare_cwt
.. autofunction:: jaxwt.apply_cwt
.. autofunction:: jaxwt.wavefun
.. autofunction:: jaxwt.integrate_wavelet
.. autofunction:: jaxwt.central_frequency
.. autofunction:: jaxwt.scale2frequency

.. autoclass:: jaxwt.ContinuousWavelet
   :members:

.. autoclass:: jaxwt.CWTKernelBank
   :members:

Fully Separable DWT
-------------------

.. autofunction:: jaxwt.fswavedecn
.. autofunction:: jaxwt.fswaverecn

.. autoclass:: jaxwt.FswavedecnResult
   :members:

Multiresolution Analysis
------------------------

.. autofunction:: jaxwt.mra
.. autofunction:: jaxwt.imra
.. autofunction:: jaxwt.mra2
.. autofunction:: jaxwt.imra2
.. autofunction:: jaxwt.mran
.. autofunction:: jaxwt.imran

Wavelet Packets
---------------

.. autofunction:: jaxwt.wp_decompose
.. autofunction:: jaxwt.wp_reconstruct
.. autofunction:: jaxwt.wp_decompose_nd
.. autofunction:: jaxwt.wp_reconstruct_nd

Thresholding
------------

.. autofunction:: jaxwt.threshold
.. autofunction:: jaxwt.threshold_firm

Utilities
---------

.. autofunction:: jaxwt.dwt_max_level
.. autofunction:: jaxwt.downcoef
.. autofunction:: jaxwt.upcoef
.. autofunction:: jaxwt.qmf
.. autofunction:: jaxwt.orthogonal_filter_bank
.. autofunction:: jaxwt.get_wavelet
.. autofunction:: jaxwt.swt_max_level

.. autoclass:: jaxwt.Wavelet
   :members:
