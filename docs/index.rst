jaxwt
=====

JAX-native wavelet transforms. Differentiable, JIT-compilable, GPU-ready.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api


Installation
------------

.. code-block:: bash

   pip install jax jaxlib jaxwt


Quick Start
-----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import jaxwt

   # Decompose
   x = jnp.ones((64, 64))
   coeffs = jaxwt.wavedecn(x, 'db4', level=3)

   # Reconstruct
   rec = jaxwt.waverecn(coeffs, 'db4')

   # Batch via vmap
   from functools import partial
   batch_transform = jax.vmap(partial(jaxwt.wavedecn, wavelet='db4', level=3))

   # Differentiate
   grad = jax.grad(lambda x: jnp.sum(jaxwt.waverecn(jaxwt.wavedecn(x, 'db4'), 'db4')))(x)
