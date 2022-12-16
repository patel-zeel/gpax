import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# jax 64 bit mode
jax.config.update("jax_enable_x64", True)

from gpax.models import LatentGPHeinonen, LatentGPDeltaInducing
from gpax.core import set_default_jitter, get_default_jitter
from gpax.models import ExactGPRegression, SparseGPRegression
from gpax.kernels import RBF
from gpax.likelihoods import Gaussian
from gpax.means import Scalar, Average
from tests.utils import assert_same_pytree, assert_approx_same_pytree

from stheno.jax import EQ, GP, OneMean, PseudoObs
import lab.jax as B


X_inducing = jax.random.uniform(jax.random.PRNGKey(0), (5, 3))
X = jax.random.uniform(jax.random.PRNGKey(1), (10, 3))
y = jax.random.normal(jax.random.PRNGKey(2), (10,))
X_new = jax.random.uniform(jax.random.PRNGKey(3), (15, 3))


@pytest.mark.parametrize("lgp_type", [LatentGPDeltaInducing, LatentGPHeinonen])
@pytest.mark.parametrize(
    "vmap, train_shapes, eval_shapes",
    [
        (False, [(X.shape[0],), ()], [(X.shape[0],), (X_new.shape[0],)]),
        (True, [X.shape, (X.shape[1],)], [X.shape, X_new.shape]),
    ],
)
def test_latent_gp(lgp_type, vmap, train_shapes, eval_shapes):
    if lgp_type is LatentGPHeinonen:
        X_inducing_tmp = X
    else:
        X_inducing_tmp = X_inducing
    lgp = lgp_type(X=X_inducing_tmp, latent_kernel_type=RBF, vmap=vmap)
    out_X, log_prior = lgp(X_inducing_tmp, X)
    assert out_X.shape == train_shapes[0]
    assert log_prior.shape == train_shapes[1]
    lgp.eval()
    out_X, out_X_new = lgp(X_inducing_tmp, X, X_new)
    assert out_X.shape == eval_shapes[0]
    assert out_X_new.shape == eval_shapes[1]


n_tests = 5
keys = [key for key in jax.random.split(jax.random.PRNGKey(10), n_tests)]


def stheno_log_prob(params):
    stheno_gp = GP(
        params["mean"]["value"] * OneMean(),
        params["kernel"]["scale"] ** 2 * EQ().stretch(params["kernel"]["lengthscale"]),
    )
    return stheno_gp(X, params["likelihood"]["scale"] ** 2).logpdf(y)


def test_init():
    ls = jnp.array(1.5).repeat(X.shape[1])
    scale = jnp.array(2.0)
    noise_scale = jnp.array(0.1)

    kernel = RBF(X, lengthscale=ls, scale=scale)
    gp = ExactGPRegression(kernel=kernel, likelihood=Gaussian(scale=noise_scale), mean=Scalar())

    values = gp.get_parameters()
    assert jnp.allclose(values["kernel"]["lengthscale"], ls)
    assert jnp.allclose(values["kernel"]["scale"], scale)
    assert jnp.allclose(values["likelihood"]["scale"], noise_scale)

    raw_values = gp.get_raw_parameters()
    assert jnp.allclose(raw_values["kernel"]["lengthscale"], gp.kernel.lengthscale.bijector.inverse(ls))
    assert jnp.allclose(raw_values["kernel"]["scale"], gp.kernel.scale.bijector.inverse(scale))
    assert jnp.allclose(raw_values["likelihood"]["scale"], gp.likelihood.scale.bijector.inverse(noise_scale))


@pytest.mark.parametrize("key", keys)
@pytest.mark.parametrize(
    "kernel",
    [
        RBF,
        # GibbsKernel(flex_scale=False, flex_variance=False),
    ],
)
def test_exact_gp(key, kernel):
    gp = ExactGPRegression(kernel=kernel(X), likelihood=Gaussian(), mean=Scalar())
    gp.initialize(key)
    log_prob = gp.log_probability(X, y)

    params = gp.get_parameters()
    assert jnp.allclose(log_prob, stheno_log_prob(params), atol=1e-4)

    # jittable
    def neg_log_prob(raw_params):
        gp = ExactGPRegression(kernel=kernel(X), likelihood=Gaussian(), mean=Scalar())
        gp.set_parameters(raw_params)
        return -gp.log_probability(X, y)

    def neg_log_prob_stheno(raw_params):
        gp = ExactGPRegression(kernel=kernel(X), likelihood=Gaussian(), mean=Scalar())
        gp.set_parameters(raw_params)
        params = gp.get_parameters()
        return -stheno_log_prob(params)

    raw_params = gp.get_parameters()
    grads = jax.jit(jax.grad(neg_log_prob))(raw_params)
    grads_stheno = jax.jit(jax.grad(neg_log_prob_stheno))(raw_params)
    assert_approx_same_pytree(grads, grads_stheno)


@pytest.mark.parametrize("seed", list(range(5)))
def test_sparse_gp_log_prob(seed):
    set_default_jitter(B.epsilon)
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (10, 1))
    key = jax.random.split(key, 1)[0]
    y = jax.random.normal(key, (10,))
    key = jax.random.split(key, 1)[0]
    X_inducing = jax.random.normal(key, (5, 1))
    gp = SparseGPRegression(RBF(X), Gaussian(), Scalar(), X_inducing)
    gp.initialize(key)
    params = gp.get_parameters()

    stheno_gp = GP(
        params["mean"]["value"] * OneMean(),
        params["kernel"]["scale"] ** 2 * EQ().stretch(params["kernel"]["lengthscale"]),
    )

    stheno_log_prob = PseudoObs(stheno_gp(X_inducing), (stheno_gp(X, params["likelihood"]["scale"] ** 2), y)).elbo(
        stheno_gp.measure
    )

    log_prob = gp.log_probability(X, y)

    assert jnp.allclose(log_prob, stheno_log_prob)


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
