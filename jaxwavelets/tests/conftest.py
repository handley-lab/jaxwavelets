"""Pytest configuration for jaxwavelets tests."""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
