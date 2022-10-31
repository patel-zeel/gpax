import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
from gpax.bijectors import Exp, Log, Identity, Softplus

import pytest

key = jax.random.PRNGKey(0)
samples = [sample for sample in 10 * jax.random.normal(key, (20,))]


@pytest.mark.parametrize("sample", samples)
@pytest.mark.parametrize("bijector", [Exp(), Log(), Identity(), Softplus()])
def test_bijector(sample, bijector):
    transformed_sample = bijector.forward(sample)
    if bijector.lower < transformed_sample < bijector.upper:
        assert bijector.inverse(transformed_sample) == pytest.approx(sample)
