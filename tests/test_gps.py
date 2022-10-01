import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

import jax
import jax.numpy as jnp
from gpax import ExactGP, GibbsKernel
from stheno.jax import EQ, GP, OneMean

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 2)
X = jax.random.normal(keys[0], (10, 3))
y = jax.random.normal(keys[1], (10,))


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_exact_gp(seed):
    key = jax.random.PRNGKey(seed)
    gp = ExactGP()
    params = gp.initialise_params(key, X=X)
    log_prob = gp.log_probability(params, X, y)

    stheno_gp = GP(
        params["mean"]["value"] * OneMean(),
        params["kernel"]["variance"] * EQ().stretch(params["kernel"]["lengthscale"]),
    )
    logpdf = stheno_gp(X, params["noise"]["variance"]).logpdf(y)

    assert jnp.allclose(log_prob, logpdf)


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_gibbs_gp(seed):
    key = jax.random.PRNGKey(seed)
    gp = ExactGP(kernel=GibbsKernel(flex_scale=False, flex_variance=False))
    params = gp.initialise_params(key, X=X)
    log_prob = gp.log_probability(params, X, y)

    stheno_gp = GP(
        params["mean"]["value"] * OneMean(),
        params["kernel"]["variance"] * EQ().stretch(params["kernel"]["lengthscale"]),
    )
    logpdf = stheno_gp(X, params["noise"]["variance"]).logpdf(y)

    assert jnp.allclose(log_prob, logpdf)
