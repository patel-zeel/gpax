import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from gpax.means import Scalar, Average
from gpax.core import Parameter
from tests.utils import assert_same_pytree


@pytest.mark.parametrize(
    "Mean, kwargs, expected, y",
    [
        (Scalar, {"value": 0.0}, 0.0, None),
        (Scalar, {"value": 1.0}, 1.0, None),
        (Average, {}, 2.0, jnp.array([1.0, 2.0, 3.0])),
    ],
)
def test_initialize(Mean, kwargs, expected, y):
    mean = Mean(**kwargs)
    value = mean(y=y)

    assert_same_pytree(value, expected)
    if isinstance(Mean, Scalar):
        assert isinstance(mean.value.bijector, tfb.Identity)


def test_functionality():
    # initialize and value check
    mean = Scalar(value=1.0)
    params = mean.get_parameters(raw_dict=False)
    assert mean.value() == 1.0
    assert params["value"]() == 1.0

    # set and value check
    mean.value.set_value(2.0)
    assert mean.value() == 2.0
    params = mean.get_parameters(raw_dict=False)
    assert params["value"]() == 2.0

    # initialize
    mean.initialize(jax.random.PRNGKey(0))
    assert mean.value() != 2.0


def test_gradients():
    mean = Scalar(value=2.0)
    params = mean.get_parameters()

    def loss_fn(params):
        mean.set_parameters(params)
        return jnp.sum(mean(y=None)) ** 3

    def manual_grad_fn(params):
        return {"value": 3 * params["value"] ** 2}

    grad_fn = jax.grad(loss_fn)
    assert_same_pytree(grad_fn(params), manual_grad_fn(params))

    # jittable
    grad_fn = jax.jit(jax.grad(loss_fn))
    assert_same_pytree(grad_fn(params), manual_grad_fn(params))
