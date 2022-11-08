import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
from gpax.bijectors import Exp, Log, Identity, SquarePlus, SoftPlus

import pytest

key = jax.random.PRNGKey(0)
samples = [sample for sample in 10 * jax.random.normal(key, (20,))]


@pytest.mark.parametrize("sample", samples)
@pytest.mark.parametrize("bijector", [Exp(), Log(), Identity(), SoftPlus(), SquarePlus()])
def test_bijector(sample, bijector):
    tol = 1e-4
    transformed_sample = bijector.forward(sample)
    if bijector.lower < transformed_sample < bijector.upper:
        if isinstance(bijector, SoftPlus):
            tol = 1e-2
        assert bijector.inverse(transformed_sample) == pytest.approx(sample, abs=tol)
