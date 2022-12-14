import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.tree_util as jtu
import jax.numpy as jnp

import pytest
import gpax.distributions as gd
import gpax.kernels as gpk
from gpax.models import ExactGPRegression
from gpax.core import Parameter
import gpax.distributions as gd
import gpax.bijectors as gb
import gpax.likelihoods as gl
import gpax.means as gm

# from gpax.special_kernels import Gibbs
from tests.utils import assert_same_pytree

import stheno
import lab.jax as B


@pytest.mark.parametrize(
    "X", [jax.random.normal(jax.random.PRNGKey(0), (3, 1)), jax.random.normal(jax.random.PRNGKey(0), (3, 2))]
)
@pytest.mark.parametrize("ls, scale", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
@pytest.mark.parametrize(
    "kernel, stheno_kernel",
    [
        (gpk.RBF, stheno.EQ),
        (gpk.Matern12, stheno.Matern12),
        (gpk.Matern32, stheno.Matern32),
        (gpk.Matern52, stheno.Matern52),
        (gpk.Polynomial, stheno.Linear),
    ],
)
def test_execution(X, kernel, stheno_kernel, ls, scale):
    if kernel is gpk.Polynomial:
        order = 1.0
        kernel = kernel(input_dim=X.shape[1], order=order)
        params = kernel.get_constrained_params()

        stheno_kernel = stheno_kernel() + params["center"]

        ours = kernel(X, X)
        stheno_vals = stheno_kernel(X, X)

        assert jnp.allclose(ours, B.dense(stheno_vals))

    else:
        kernel = kernel(input_dim=X.shape[1], lengthscale=ls, scale=scale)
        params = kernel.get_constrained_params()
        assert len(params["lengthscale"]) == X.shape[1]
        kernel_stheno = (params["scale"] ** 2) * stheno_kernel().stretch(params["lengthscale"])
        ours = kernel(X, X)
        stheno_vals = kernel_stheno(X, X)

        assert jnp.allclose(ours, B.dense(stheno_vals), atol=1e-2)


def test_combinations():
    X = jax.random.normal(jax.random.PRNGKey(0), (3, 1))
    kernel = (
        gpk.RBF(input_dim=X.shape[1], lengthscale=0.1, scale=0.2)
        * gpk.Matern12(input_dim=X.shape[1], lengthscale=0.3, scale=0.4)
    ) * gpk.Polynomial(input_dim=X.shape[1], center=0.5, order=1.0)
    kernel_stheno = ((0.2**2 * stheno.EQ().stretch(0.1)) * (0.4**2 * stheno.Matern12().stretch(0.3))) * (
        0.5 + stheno.Linear()
    )

    ours = kernel(X, X)
    stheno_vals = kernel_stheno(X, X)

    assert jnp.allclose(ours, B.dense(stheno_vals), atol=1e-2)


def test_gibbs_kernel():
    num_datapoints = 10
    num_inducing = 3
    num_dims = 2

    X = jax.random.normal(jax.random.PRNGKey(0), (num_datapoints, num_dims))
    X_inducing = jax.random.normal(jax.random.PRNGKey(1), (num_inducing, num_dims))
    X_inducing = Parameter(X_inducing, fixed_init=True)
    ls_latent_gp = ExactGPRegression(kernel=gpk.RBF(input_dim=num_dims), mean=gm.Scalar(), likelihood=gl.Gaussian())
    s_latent_gp = ExactGPRegression(kernel=gpk.RBF(input_dim=num_dims), mean=gm.Scalar(), likelihood=gl.Gaussian())
    lenghscale = 1.0
    scale = 1.0

    kernel = gpk.Gibbs(
        input_dim=num_dims,
        lengthscale=lenghscale,
        scale=scale,
        X_inducing=X_inducing,
        lengthscale_gp=ls_latent_gp,
        scale_gp=s_latent_gp,
    )

    cov = kernel(X, X)
    assert cov.shape == (num_datapoints, num_datapoints)

    kernel = gpk.Gibbs(
        input_dim=num_dims,
        lengthscale=lenghscale,
        scale=scale,
        flex_lengthscale=False,
        flex_scale=False,
    )

    cov = kernel(X, X)
    assert cov.shape == (num_datapoints, num_datapoints)


# def test_gibbs_kernel_dry_run():
#     X_inducing = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
#     X = jax.random.normal(jax.random.PRNGKey(1), (6, 2))
#     kernel = Gibbs(X_inducing=X_inducing)
#     params = kernel.initialize_params(key=jax.random.PRNGKey(0), X_inducing=X_inducing)
#     bijectrors = kernel.get_bijectors()

#     params = constrain(params, bijectrors)


# def test_gibbs_combinations():
#     X_inducing = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
#     kernel = (RBF() + Gibbs(X_inducing=X_inducing)) * Polynomial()

#     params = kernel.initialize_params(key=jax.random.PRNGKey(0), X_inducing=X_inducing)
#     bijectrors = kernel.get_bijectors()

#     params = constrain(params, bijectrors)
