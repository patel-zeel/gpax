import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.tree_util as jtu
import jax.numpy as jnp

import pytest
import gpax.kernels as gpk
from gpax.models import ExactGPRegression, LatentGPHeinonen, LatentGPDeltaInducing
from gpax.core import Parameter
import gpax.kernels as gpk

from tests.utils import assert_same_pytree

from GPy.kern import RBF as EQ, Matern32, Matern52, Exponential, Poly


@pytest.mark.parametrize(
    "X", [jax.random.normal(jax.random.PRNGKey(0), (3, 1)), jax.random.normal(jax.random.PRNGKey(0), (2, 2))]
)
@pytest.mark.parametrize("ls, scale", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
@pytest.mark.parametrize(
    "kernel, gpy_kernel",
    [
        (gpk.RBF, EQ),
        (gpk.Matern12, Exponential),
        (gpk.Matern32, Matern32),
        (gpk.Matern52, Matern52),
        (gpk.Polynomial, Poly),
    ],
)
def test_execution(X, kernel, gpy_kernel, ls, scale):
    if kernel is gpk.Polynomial:
        order = 1.0
        kernel = kernel(X=X, order=order)
        kernel_fn = kernel.eval().get_kernel_fn()
        params = kernel.get_parameters()

        gpy_kernel = gpy_kernel(X.shape[1], bias=params["center"], order=order)

        ours = kernel_fn(X, X)
        gpy_vals = gpy_kernel.K(X, X)

        assert jnp.allclose(ours, gpy_vals)

    else:
        kernel = kernel(X=X, lengthscale=ls, scale=scale, ARD=True)
        kernel_fn = kernel.eval().get_kernel_fn()
        params = kernel.get_parameters()
        assert len(params["lengthscale"]) == X.shape[1]
        kernel_gpy = gpy_kernel(X.shape[1], lengthscale=params["lengthscale"], variance=params["scale"] ** 2, ARD=True)
        ours = kernel_fn(X, X)
        gpy_vals = kernel_gpy.K(X, X)

        assert jnp.allclose(ours, gpy_vals, atol=1e-4)


def test_combinations():
    X = jax.random.normal(jax.random.PRNGKey(0), (3, 1))
    kernel = (
        gpk.RBF(X=X, lengthscale=0.1, scale=0.2) * gpk.Matern12(X=X, lengthscale=0.3, scale=0.4)
    ) * gpk.Polynomial(X=X, center=0.5, order=1.0)
    kernel_gpy = ((EQ(X.shape[1], 0.2**2, 0.1)) * (Exponential(X.shape[1], 0.4**2, 0.3))) * (
        Poly(X.shape[1], bias=0.5, order=1.0)
    )

    ours = kernel.eval().get_kernel_fn()(X, X)
    gpy_vals = kernel_gpy.K(X, X)

    assert jnp.allclose(ours, gpy_vals)


@pytest.mark.parametrize("model", [LatentGPDeltaInducing, LatentGPHeinonen])
def test_gibbs_kernel(model):
    X_inducing = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    X = jax.random.normal(jax.random.PRNGKey(0), (6, 2))
    X_new = jax.random.normal(jax.random.PRNGKey(0), (12, 2))

    if model is LatentGPHeinonen:
        X_inducing = X

    ell_kernel = gpk.RBF(X_inducing, 0.1, 0.2)
    ell_model = model(X_inducing, ell_kernel, vmap=True)
    sigma_kernel = gpk.RBF(X_inducing, 0.3, 0.4)
    sigma_model = model(X_inducing, sigma_kernel)
    kernel = gpk.Gibbs(
        X_inducing,
        ell_model,
        sigma_model,
    )

    kernel_fn = kernel.get_kernel_fn(X_inducing)
    cov, log_prior = kernel_fn(X, X)
    assert cov.shape == (X.shape[0], X.shape[0])

    kernel_fn = kernel.eval().get_kernel_fn(X_inducing)
    cross_cov = kernel_fn(X, X_new)
    assert cross_cov.shape == (X.shape[0], X_new.shape[0])

    new_cov = kernel_fn(X_new, X_new)
    assert new_cov.shape == (X_new.shape[0], X_new.shape[0])
