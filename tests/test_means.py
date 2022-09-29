import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp

from gpax import ConstantMean
from gpax.bijectors import Identity
from tests.utils import assert_same_pytree


@pytest.mark.parametrize("value, expected", [(None, 0.0), (1.0, 1.0)])
def test_initialise(value, expected):
    mean = ConstantMean(value=value)
    key = jax.random.PRNGKey(0)
    params = mean.initialise_params(key)
    params_expected = {"mean": {"value": jnp.array(expected)}}
    assert_same_pytree(params, params_expected)

    bijectors = mean.get_bijectors()
    bijectors_expected = {"mean": {"value": Identity()}}
    assert_same_pytree(bijectors, bijectors_expected)
