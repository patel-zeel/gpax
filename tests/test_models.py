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
import gpax.kernels as gpk
from gpax.likelihoods import Gaussian
from gpax.means import Scalar, Average
from tests.utils import assert_same_pytree, assert_approx_same_pytree

import GPy
from GPy.kern import RBF, Matern32, Matern52, Exponential, Poly
from GPy.models import GPRegression, SparseGPRegression as GPySparseGPRegression

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
    kernel = gpk.RBF(X)
    lgp = lgp_type(X_inducing_tmp, kernel, vmap=vmap)
    out_X, log_prior = lgp(X_inducing_tmp)(X)
    assert out_X.shape == train_shapes[0]
    assert log_prior.shape == train_shapes[1]
    lgp.eval()
    out_X = lgp(X_inducing_tmp)(X)
    out_X_new = lgp(X_inducing_tmp)(X_new)
    assert out_X.shape == eval_shapes[0]
    assert out_X_new.shape == eval_shapes[1]


n_tests = 5
keys = [key for key in jax.random.split(jax.random.PRNGKey(10), n_tests)]


def gpy_loss(params):
    kernel = RBF(X.shape[1], params["kernel"]["variance"], params["kernel"]["base_kernel"]["lengthscale"], ARD=True)
    mean = GPy.core.Mapping(X.shape[1], 1)
    mean.f = lambda x: params["mean"]["value"]
    mean.update_gradients = lambda a, b: None
    gpy_gp = GPRegression(X, y.reshape(-1, 1), kernel, mean_function=mean)
    gpy_gp.likelihood.variance = params["likelihood"]["scale"] ** 2
    return gpy_gp.objective_function() / X.size


def gpy_loss_sparse(params):
    kernel = RBF(X.shape[1], params["kernel"]["variance"], params["kernel"]["base_kernel"]["lengthscale"], ARD=True)
    mean = GPy.core.Mapping(X.shape[1], 1)
    mean.f = lambda x: params["mean"]["value"]
    mean.update_gradients = lambda a, b: None
    gpy_gp = GPySparseGPRegression(X, y.reshape(-1, 1), kernel, mean_function=mean, Z=params["X_inducing"])
    gpy_gp.likelihood.variance = params["likelihood"]["scale"] ** 2
    return gpy_gp.objective_function() / X.size


def test_init():
    ls = jnp.array(1.5).repeat(X.shape[1])
    scale = jnp.array(2.0)
    noise_scale = jnp.array(0.1)

    kernel = gpk.Scale(X, gpk.RBF(X, lengthscale=ls), variance=scale**2)
    gp = ExactGPRegression(kernel=kernel, likelihood=Gaussian(scale=noise_scale), mean=Scalar())

    values = gp.get_parameters()
    assert jnp.allclose(values["kernel"]["base_kernel"]["lengthscale"], ls)
    assert jnp.allclose(values["kernel"]["variance"], scale**2)
    assert jnp.allclose(values["likelihood"]["scale"], noise_scale)

    raw_values = gp.get_raw_parameters()
    assert jnp.allclose(
        raw_values["kernel"]["base_kernel"]["lengthscale"], gp.kernel.base_kernel.lengthscale.bijector.inverse(ls)
    )
    assert jnp.allclose(raw_values["kernel"]["variance"], gp.kernel.variance.bijector.inverse(scale**2))
    assert jnp.allclose(raw_values["likelihood"]["scale"], gp.likelihood.scale.bijector.inverse(noise_scale))


@pytest.mark.parametrize("key", keys)
@pytest.mark.parametrize(
    "base_kernel_fn",
    [
        gpk.RBF,
        # GibbsKernel(flex_scale=False, flex_variance=False),
    ],
)
def test_exact_gp(key, base_kernel_fn):
    base_kernel = base_kernel_fn(X, lengthscale=0.1)
    kernel = gpk.Scale(X, base_kernel, variance=0.2)
    gp = ExactGPRegression(kernel=kernel, likelihood=Gaussian(), mean=Scalar())
    gp.initialize(key)
    log_prob = gp.log_probability(X, y)

    params = gp.get_parameters()
    assert jnp.allclose(log_prob, -gpy_loss(params), atol=1e-4)

    # jittable
    def neg_log_prob(raw_params):
        gp = ExactGPRegression(kernel=kernel, likelihood=Gaussian(), mean=Scalar())
        gp.set_parameters(raw_params)
        return -gp.log_probability(X, y)

    raw_params = gp.get_raw_parameters()
    grads = jax.jit(jax.grad(neg_log_prob))(raw_params)

    gp.set_raw_parameters(raw_params)

    # predict
    pred_mean, pred_var = gp.predict(X, y, X_new, full_cov=False)
    assert pred_mean.shape == (X_new.shape[0],)
    assert pred_var.shape == (X_new.shape[0],)
    pred_mean, pred_var = gp.predict(X, y, X_new, full_cov=True)
    assert pred_mean.shape == (X_new.shape[0],)
    assert pred_var.shape == (X_new.shape[0], X_new.shape[0])

    # condition and predict
    predict_fn = gp.condition(X, y)
    pred_mean, pred_var = predict_fn(X_new, full_cov=False)
    assert pred_mean.shape == (X_new.shape[0],)
    assert pred_var.shape == (X_new.shape[0],)
    pred_mean, pred_var = predict_fn(X_new, full_cov=True)
    assert pred_mean.shape == (X_new.shape[0],)
    assert pred_var.shape == (X_new.shape[0], X_new.shape[0])


@pytest.mark.parametrize("seed", list(range(5)))
def test_sparse_gp(seed):
    key = jax.random.PRNGKey(seed)
    gp = SparseGPRegression(gpk.Scale(X, gpk.RBF(X)), Gaussian(), Scalar(), X_inducing)
    gp.initialize(key)
    params = gp.get_parameters()

    gpy_loss = gpy_loss_sparse(params)

    log_prob = gp.log_probability(X, y)

    assert jnp.allclose(log_prob, -gpy_loss, atol=1e-4)


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

#     gpy_gp = GP(
#         params["mean"]["value"] * OneMean(),
#         params["kernel"]["variance"] * EQ().stretch(params["kernel"]["lengthscale"]),
#     )

#     pseudo_obs = PseudoObs(gpy_gp(X_inducing), (gpy_gp(X, params["noise"]["variance"]), y))
#     f_post = gpy_gp | pseudo_obs

#     pred_mean_gpy = B.dense(f_post(X_test).mean).squeeze()
#     pred_cov_gpy = B.dense(f_post(X_test).var)

#     pred_mean, pred_cov = gp.predict(params, X, y, X_test, return_cov=True, include_noise=False)

#     assert jnp.allclose(pred_mean, pred_mean_gpy)
#     assert jnp.allclose(pred_cov, pred_cov_gpy)
