import pytest
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp

from gpax import ConstantMean
from tests.utils import assert_same_pytree

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors


@pytest.mark.parametrize("value, expected", [(None, 0.0), (1.0, 1.0)])
def test_initialise(value, expected):
    mean = ConstantMean(value=value)
    key = jax.random.PRNGKey(0)
    params = mean.initialise_params(key)
    params_expected = {"mean": {"value": jnp.array(expected)}}
    assert_same_pytree(params, params_expected)

    bijectors = mean.get_bijectors()
    bijectors_expected = {"mean": {"value": tfb.Identity()}}
    assert_same_pytree(bijectors, bijectors_expected)
