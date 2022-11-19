import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import jax
import jax.tree_util as jtu
import jax.numpy as jnp

from gpax.means import Scalar, Average
from gpax.core import Parameter
import gpax.bijectors as gb
import gpax.distributions as gd
from tests.utils import assert_same_pytree


@pytest.mark.parametrize(
    "Mean, kwargs, expected, bijectors_expected, y",
    [
        (Scalar, {"value": Parameter(0.0)}, 0.0, gb.get_default_bijector(), None),
        (Scalar, {"value": Parameter(1.0)}, 1.0, gb.get_default_bijector(), None),
        (Average, {}, 2.0, {}, jnp.array([1.0, 2.0, 3.0])),
    ],
)
def test_initialize(Mean, kwargs, expected, bijectors_expected, y):
    mean = Mean(**kwargs)
    value = mean(y=y)

    assert_same_pytree(value, expected)
    if isinstance(Mean, Scalar):
        assert_same_pytree(mean.get_params()["value"]._priors, bijectors_expected)


def test_functionality():
    # initialize and value check
    mean = Scalar(value=1.0)
    params = mean.get_params(raw_dict=False)
    assert mean.value() == 1.0
    assert params["value"]() == 1.0
    assert id(mean.value) == id(params["value"])

    # set and value check
    mean.value.set(2.0)
    assert mean.value() == 2.0
    assert params["value"]() == 2.0

    # initialize
    mean.initialize(jax.random.PRNGKey(0))


def test_gradients():
    mean = Scalar(value=2.0)
    params = mean.get_params()

    def loss_fn(params):
        mean.set_params(params)
        return jnp.sum(mean()) ** 3

    def manual_grad_fn(params):
        return {"value": 3 * params["value"] ** 2}

    grad_fn = jax.grad(loss_fn)
    assert_same_pytree(grad_fn(params), manual_grad_fn(params))

    # jittable
    grad_fn = jax.jit(jax.grad(loss_fn))
    assert_same_pytree(grad_fn(params), manual_grad_fn(params))
