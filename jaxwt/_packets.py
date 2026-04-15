"""Wavelet packet decomposition and reconstruction."""
import jax.numpy as jnp
from jaxwt._dwt import dwt, idwt, dwt_max_level
from jaxwt._multidim import dwtn, idwtn
from jaxwt._filters import get_wavelet


def wp_decompose(data, wavelet, mode='symmetric', maxlevel=None):
    """1D wavelet packet decomposition. Returns (leaves, shapes).

    leaves: dict mapping path strings to arrays (all paths have length maxlevel)
    shapes: dict mapping parent paths to their data shapes (for reconstruction trimming)
    """
    w = get_wavelet(wavelet)
    if maxlevel is None:
        maxlevel = dwt_max_level(data.shape[0], w.dec_lo.shape[0])
    leaves = {'': data}
    shapes = {}
    for _ in range(maxlevel):
        new_leaves = {}
        for path, x in leaves.items():
            shapes[path] = x.shape
            cA, cD = dwt(x, w, mode)
            new_leaves[path + 'a'] = cA
            new_leaves[path + 'd'] = cD
        leaves = new_leaves
    return leaves, shapes


def wp_reconstruct(leaves, wavelet, mode='symmetric'):
    """1D wavelet packet reconstruction from complete leaf set.

    Expects all 2^maxlevel leaves at the same depth.
    """
    w = get_wavelet(wavelet)
    nodes = dict(leaves)
    maxlevel = max(len(k) for k in nodes)
    for level in range(maxlevel, 0, -1):
        new_nodes = {}
        for path, arr in sorted(nodes.items()):
            if len(path) == level:
                parent = path[:-1]
                new_nodes.setdefault(parent, {})[path[-1]] = arr
        for parent, children in new_nodes.items():
            nodes[parent] = idwt(children['a'], children['d'], w, mode)
        # Remove consumed leaves
        nodes = {k: v for k, v in nodes.items() if len(k) < level}
    return nodes['']


def wp_decompose_nd(data, wavelet, mode='symmetric', maxlevel=None, axes=None):
    """nD wavelet packet decomposition. Returns (leaves, shapes)."""
    if axes is None:
        axes = tuple(range(data.ndim))
    else:
        axes = tuple(axes)
    w = get_wavelet(wavelet)
    ndim = len(axes)
    if maxlevel is None:
        maxlevel = dwt_max_level(min(data.shape[ax] for ax in axes), w.dec_lo.shape[0])
    leaves = {'': data}
    shapes = {}
    for _ in range(maxlevel):
        new_leaves = {}
        for path, x in leaves.items():
            shapes[path] = x.shape
            subbands = dwtn(x, w, mode, axes)
            for key, arr in subbands.items():
                new_leaves[path + key] = arr
        leaves = new_leaves
    return leaves, shapes


def wp_reconstruct_nd(leaves, wavelet, mode='symmetric', axes=None, ndim_transform=None):
    """nD wavelet packet reconstruction from complete leaf set."""
    axes = tuple(axes) if axes is not None else None
    w = get_wavelet(wavelet)
    nodes = dict(leaves)
    sample_key = next(iter(nodes))
    # Infer ndim from axes or ndim_transform
    ndim = len(axes) if axes is not None else ndim_transform
    maxlevel = len(sample_key) // ndim
    for level in range(maxlevel, 0, -1):
        key_len = level * ndim
        new_nodes = {}
        for path, arr in sorted(nodes.items()):
            if len(path) == key_len:
                parent = path[:key_len - ndim]
                subband = path[key_len - ndim:]
                new_nodes.setdefault(parent, {})[subband] = arr
        for parent, children in new_nodes.items():
            nodes[parent] = idwtn(children, w, mode, axes)
        nodes = {k: v for k, v in nodes.items() if len(k) < key_len}
    return nodes['']
