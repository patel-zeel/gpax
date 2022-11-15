import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import jax
import jax.tree_util as jtu
import jax.numpy as jnp

from gpax.means import Scalar, Average
from gpax.core import get_default_bijector
from tests.utils import assert_same_pytree


@pytest.mark.parametrize(
    "Mean, kwargs, expected, bijectors_expected",
    [
        (Scalar, {"value": 0.0}, 0.0, {"value": get_default_bijector()}),
        (Scalar, {"value": 1.0}, 1.0, {"value": get_default_bijector()}),
        (Average, {}, 2.0, {}),
    ],
)
def test_initialize(Mean, kwargs, expected, bijectors_expected):
    mean = Mean(**kwargs)
    value = mean(aux={"y": jnp.array([1.0, 2.0, 3.0])})

    assert_same_pytree(value, expected)
    assert_same_pytree(mean.constraints, bijectors_expected)


def test_flatten():
    mean = Scalar(value=1.0)
    params = mean.get_params()
    params["value"] = 2.0
    mean.set_params(params)
    assert mean.value == 2.0

    values, treedef = jtu.tree_flatten(mean)
    assert values == [2.0]
    values = (3.0,)
    mean = jtu.tree_unflatten(treedef, values)
    assert mean.value == 3.0

    mean.constrain()
    assert mean.value == 3.0
    mean.unconstrain()
    assert mean.value == 3.0
