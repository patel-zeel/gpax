import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import gpax.bijectors as gb
import gpax.distributions as gd

import pytest

key = jax.random.PRNGKey(1)
samples = [sample for sample in 10 * jax.random.normal(key, (5,))]


@pytest.mark.parametrize("sample", samples)
@pytest.mark.parametrize("bijector", [gb.Exp(), gb.Identity(), gb.Softplus(), gb.SquarePlus()])
def test_value_transform(sample, bijector):
    tol = 0.01
    lower, upper = bijector.in_limits()
    transformed_sample = bijector.forward(sample)
    lower, upper = bijector.out_limits()
    assert lower <= transformed_sample < upper
    re_inverse = bijector.inverse(transformed_sample)
    if isinstance(bijector, (gb.Softplus, gb.SquarePlus)):
        if re_inverse == -jnp.inf:
            if sample < -18:
                assert True
            else:
                assert False
        else:
            assert jnp.allclose(sample, re_inverse, atol=tol)
    else:
        assert jnp.allclose(sample, re_inverse, atol=tol)


@pytest.mark.parametrize("bijector", [gb.Exp(), gb.Log(), gb.Identity(), gb.Softplus(), gb.SquarePlus(), gb.White()])
def test_dist_transform(bijector):
    dist = gd.Normal(loc=0.0, scale=1.0)
    trans_dist = bijector(dist)
    re_trans_dist = bijector.inverse(trans_dist)
    assert dist.__class__ is re_trans_dist.__class__

    inv_dist = bijector.inverse(dist)
    re_trans_dist = bijector(inv_dist)
    assert dist.__class__ is re_trans_dist.__class__
