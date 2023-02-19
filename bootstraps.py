import pandas as pd
import numpy as np
from numpy.random import default_rng
from typing import Optional


def _id_nb_bootstrap(
    n_obs: int, block_length: int, random_seed: Optional[int] = 10
) -> np.ndarray:
    """Create bootstrapped indexes with the none overlapping block bootstrap
    ('nbb') strategy given the number of observations in a timeseries and
    the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """
    rng = default_rng(random_seed)

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    blocks = rng.permutation(x=np.arange(0, n_obs, block_length)).reshape(-1, 1)

    _id = (blocks + nexts).ravel()[:n_obs]

    return _id


def _id_mb_bootstrap(
    n_obs: int, block_length: int, random_seed: Optional[int] = 10
) -> np.ndarray:
    """Create bootstrapped indexes with the moving block bootstrap
    ('mbb') strategy given the number of observations in a timeseries
    and the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """
    rng = default_rng(random_seed)
    try:
        rng_integers = rng.integers
    except AttributeError:
        rng_integers = rng.randint

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    last_block = n_obs - block_length
    blocks = rng_integers(low=0, high=last_block, size=(n_blocks, 1), dtype=int)
    _id = (blocks + nexts).ravel()[:n_obs]

    return _id


def _id_cb_bootstrap(
    n_obs: int, block_length: int, random_seed: Optional[int] = 10
) -> np.ndarray:
    """Create bootstrapped indexes with the circular block bootstrap
    ('cbb') strategy given the number of observations in a timeseries
    and the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """
    rng = default_rng(random_seed)
    try:
        rng_integers = rng.integers
    except AttributeError:
        rng_integers = rng.randint

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    last_block = n_obs

    blocks = rng_integers(low=0, high=last_block, size=(n_blocks, 1), dtype=int)
    _id = np.mod((blocks + nexts).ravel(), n_obs)[:n_obs]

    return _id


def _id_s_bootstrap(
    n_obs: int, block_length: int, random_seed: Optional[int] = 10
) -> np.ndarray:
    """Create bootstrapped indexes with the stationary bootstrap
    ('sb') strategy given the number of observations in a timeseries
    and the length of the blocks.
    Returns
    -------
    _id : array
        Bootstrapped indexes.
    """
    rng = default_rng(random_seed)
    try:
        rng_integers = rng.integers
    except AttributeError:
        rng_integers = rng.randint
    #
    rng_poisson = rng.poisson
    #
    random_block_length = rng_poisson(lam=block_length, size=n_obs)
    random_block_length[random_block_length < 3] = 3
    random_block_length[random_block_length >= n_obs] = n_obs
    random_block_length = random_block_length[random_block_length.cumsum() <= n_obs]
    residual_block = n_obs - random_block_length.sum()
    if residual_block > 0:
        random_block_length = np.append(random_block_length, residual_block)
    #
    n_blocks = random_block_length.shape[0]
    nexts = np.zeros((n_blocks, random_block_length.max() + 1))
    nexts[np.arange(n_blocks), random_block_length] = 1
    nexts = np.flip(nexts, 1).cumsum(1).cumsum(1).ravel()
    nexts = (nexts[nexts > 1] - 2).astype(int)
    #
    last_block = n_obs - random_block_length.max()
    blocks = np.zeros(n_obs, dtype=int)
    if last_block > 0:
        blocks = rng_integers(low=0, high=last_block, size=n_blocks)
        blocks = np.repeat(blocks, random_block_length)
    _id = blocks + nexts
    #
    return _id
