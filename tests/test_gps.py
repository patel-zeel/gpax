import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

import jax
import jax.numpy as jnp
from gpax import ExactGP, GibbsKernel, RBFKernel, SparseGP
from stheno.jax import EQ, GP, OneMean, PseudoObs
import lab.jax as B

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 2)
X = jax.random.normal(keys[0], (10, 3))
y = jax.random.normal(keys[1], (10,))

keys = [key for key in jax.random.split(keys[-1], 3)]


def stheno_log_prob(params):
    stheno_gp = GP(
        params["mean"]["value"] * OneMean(),
        params["kernel"]["variance"] * EQ().stretch(params["kernel"]["lengthscale"]),
    )
    return stheno_gp(X, params["noise"]["variance"]).logpdf(y)


@pytest.mark.parametrize("key", keys)
@pytest.mark.parametrize("kernel", [RBFKernel(), GibbsKernel(flex_scale=False, flex_variance=False)])
def test_exact_gp(key, kernel):
    gp = ExactGP(kernel=kernel)
    params = gp.initialise_params(key, X=X)
    log_prob = gp.log_probability(params, X, y)

    assert jnp.allclose(log_prob, stheno_log_prob(params))


@pytest.mark.parametrize("seed", list(range(5)))
def test_sparse_gp(seed):
    jnp.jitter = B.epsilon
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (10, 1))
    key = jax.random.split(key, 1)[0]
    y = jax.random.normal(key, (10,))
    key = jax.random.split(key, 1)[0]
    X_inducing = jax.random.normal(key, (5, 1))
    gp = SparseGP(X_inducing=X_inducing)
    params = gp.initialise_params(key, X=X, X_inducing=X_inducing)

    stheno_gp = GP(
        params["mean"]["value"] * OneMean(),
        params["kernel"]["variance"] * EQ().stretch(params["kernel"]["lengthscale"]),
    )

    stheno_log_prob = -PseudoObs(stheno_gp(X_inducing), (stheno_gp(X, params["noise"]["variance"]), y)).elbo(
        stheno_gp.measure
    )

    log_prob = gp.log_probability(params, X, y)

    assert jnp.allclose(log_prob, stheno_log_prob)
