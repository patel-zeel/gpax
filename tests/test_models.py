import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

import jax
import jax.numpy as jnp
from gpax.models import ExactGPRegression
from gpax.kernels import RBF
from gpax.likelihoods import Gaussian
from gpax.means import Scalar

# from gpax.special_kernels import GibbsKernel
from stheno.jax import EQ, GP, OneMean, PseudoObs
import lab.jax as B

# jax 64 bit mode
jax.config.update("jax_enable_x64", True)

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
    return stheno_gp(X, params["likelihood"]["variance"]).logpdf(y)


@pytest.mark.parametrize("key", keys)
@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        # GibbsKernel(flex_scale=False, flex_variance=False),
    ],
)
def test_exact_gp(key, kernel):
    gp = ExactGPRegression(kernel=kernel, likelihood=Gaussian(), mean=Scalar())
    params = gp.initialize_params(key, aux={"X": X})
    log_prob = gp.log_probability(params, X, y)

    assert jnp.allclose(log_prob, stheno_log_prob(params))


# @pytest.mark.parametrize("seed", list(range(5)))
# def test_sparse_gp_log_prob(seed):
#     jnp.jitter = B.epsilon
#     key = jax.random.PRNGKey(seed)
#     X = jax.random.normal(key, (10, 1))
#     key = jax.random.split(key, 1)[0]
#     y = jax.random.normal(key, (10,))
#     key = jax.random.split(key, 1)[0]
#     X_inducing = jax.random.normal(key, (5, 1))
#     gp = SparseGP(X_inducing=X_inducing)
#     params = gp.initialize_params(key, X=X, X_inducing=X_inducing)

#     stheno_gp = GP(
#         params["mean"]["value"] * OneMean(),
#         params["kernel"]["variance"] * EQ().stretch(params["kernel"]["lengthscale"]),
#     )

#     stheno_log_prob = -PseudoObs(stheno_gp(X_inducing), (stheno_gp(X, params["noise"]["variance"]), y)).elbo(
#         stheno_gp.measure
#     )

#     log_prob = gp.log_probability(params, X, y)

#     assert jnp.allclose(log_prob, stheno_log_prob)


# @pytest.mark.parametrize("seed", list(range(5)))
# def test_sparse_gp_prediction(seed):
#     jnp.jitter = B.epsilon
#     key = jax.random.PRNGKey(seed)
#     X = jax.random.normal(key, (10, 1))
#     key = jax.random.split(key, 1)[0]
#     X_test = jax.random.normal(key, (15, 1))
#     key = jax.random.split(key, 1)[0]
#     y = jax.random.normal(key, (10,))
#     key = jax.random.split(key, 1)[0]
#     X_inducing = jax.random.normal(key, (5, 1))
#     gp = SparseGP(X_inducing=X_inducing)
#     params = gp.initialize_params(key, X=X, X_inducing=X_inducing)

#     stheno_gp = GP(
#         params["mean"]["value"] * OneMean(),
#         params["kernel"]["variance"] * EQ().stretch(params["kernel"]["lengthscale"]),
#     )

#     pseudo_obs = PseudoObs(stheno_gp(X_inducing), (stheno_gp(X, params["noise"]["variance"]), y))
#     f_post = stheno_gp | pseudo_obs

#     pred_mean_stheno = B.dense(f_post(X_test).mean).squeeze()
#     pred_cov_stheno = B.dense(f_post(X_test).var)

#     pred_mean, pred_cov = gp.predict(params, X, y, X_test, return_cov=True, include_noise=False)

#     assert jnp.allclose(pred_mean, pred_mean_stheno)
#     assert jnp.allclose(pred_cov, pred_cov_stheno)
