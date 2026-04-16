"""Pytest configuration for jaxwt tests."""
import jax

jax.config.update("jax_enable_x64", True)
