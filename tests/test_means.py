import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import jax
import jax.numpy as jnp

from gpax.means import ScalarMean, ZeroMean
from gpax.core import get_default_bijector
from tests.utils import assert_same_pytree


@pytest.mark.parametrize(
    "Mean, kwargs, expected, bijectors_expected",
    [
        (ScalarMean, {"value": 0.0}, {"value": jnp.zeros(())}, {"value": get_default_bijector()}),
        (ScalarMean, {"value": 1.0}, {"value": jnp.ones(())}, {"value": get_default_bijector()}),
        (ZeroMean, {}, {}, {}),
    ],
)
def test_initialize(Mean, kwargs, expected, bijectors_expected):
    mean = Mean(**kwargs)
    params = mean.initialize_params(aux={"y": jnp.array([1.0, 2.0, 3.0])})

    assert_same_pytree(params, expected)
    assert_same_pytree(mean.constraints, bijectors_expected)
