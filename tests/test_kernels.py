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

from GPy.kern import RBF as EQ, Matern32, Matern52, Exponential, Poly, StdPeriodic, RatQuad

# jax 64 bit mode
jax.config.update("jax_enable_x64", True)


def test_active_dims():
    x = jnp.ones((3, 4)) * jnp.nan
    x = x.at[:, 2].set(jnp.arange(3))

    kernel = gpk.RBF(x, active_dims=[2])
    K = kernel.eval().get_kernel_fn()(x, x)
    assert K.sum() is not jnp.nan

    for each in [0, 1, 3]:
        kernel = gpk.RBF(x, active_dims=[each])
        K = kernel.eval().get_kernel_fn()(x, x)
        assert K.sum() is jnp.nan


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
        (gpk.RationalQuadratic, RatQuad),
        (gpk.Polynomial, Poly),
        (gpk.Periodic, StdPeriodic),
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

    elif kernel is gpk.RationalQuadratic:
        alpha = 2.7
        base_kernel = kernel(X=X, alpha=alpha, lengthscale=ls)
        kernel = gpk.Scale(X, base_kernel, variance=scale**2)
        kernel_fn = kernel.eval().get_kernel_fn()
        params = kernel.get_parameters()

        gpy_kernel = gpy_kernel(X.shape[1], power=alpha, lengthscale=ls, variance=scale**2)

        ours = kernel_fn(X, X)
        gpy_vals = gpy_kernel.K(X, X)

        assert jnp.allclose(ours, gpy_vals)

    elif kernel is gpk.Periodic:
        if X.shape[1] > 1:
            pytest.skip("Periodic kernel is only implemented for 1D inputs")
        # TODO: fix this test and the periodic kernel
        period = 2.3
        base_kernel = kernel(X=X, period=period, lengthscale=ls)
        kernel = gpk.Scale(X, base_kernel, variance=scale**2)
        kernel_fn = kernel.eval().get_kernel_fn()
        params = kernel.get_parameters()

        gpy_kernel = gpy_kernel(X.shape[1], variance=scale**2, lengthscale=ls, period=period)

        ours = kernel_fn(X, X)
        gpy_vals = gpy_kernel.K(X, X)

        assert jnp.allclose(ours, gpy_vals, atol=1e-4)

    else:
        base_kernel = kernel(X, lengthscale=ls, ARD=True)
        kernel = gpk.Scale(X, base_kernel, variance=scale**2)
        kernel_fn = kernel.eval().get_kernel_fn()
        params = kernel.get_parameters()
        assert len(params["base_kernel"]["lengthscale"]) == X.shape[1]
        kernel_gpy = gpy_kernel(X.shape[1], lengthscale=ls, variance=scale**2, ARD=True)
        ours = kernel_fn(X, X)
        gpy_vals = kernel_gpy.K(X, X)

        assert jnp.allclose(ours, gpy_vals, atol=1e-4)


def test_combinations():
    X = jax.random.normal(jax.random.PRNGKey(0), (3, 1))
    kernel = (
        gpk.Scale(X, gpk.RBF(X, lengthscale=0.1), variance=0.2**2)
        * gpk.Scale(X, gpk.Matern12(X=X, lengthscale=0.3), variance=0.4**2)
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

    ell_kernel = gpk.Scale(X_inducing, gpk.RBF(X_inducing, 0.1), 0.2)
    ell_model = model(X_inducing, ell_kernel, vmap=True)
    sigma_kernel = gpk.Scale(X_inducing, gpk.RBF(X_inducing, 0.3), 0.4)
    sigma_model = model(X_inducing, sigma_kernel)
    base_kernel = gpk.Gibbs(
        X_inducing,
        ell_model,
    )
    kernel = gpk.InputDependentScale(X_inducing, base_kernel, sigma_model)

    kernel_fn = kernel.get_kernel_fn(X_inducing)
    cov, log_prior = kernel_fn(X, X)
    assert cov.shape == (X.shape[0], X.shape[0])

    kernel_fn = kernel.eval().get_kernel_fn(X_inducing)
    cross_cov = kernel_fn(X, X_new)
    assert cross_cov.shape == (X.shape[0], X_new.shape[0])

    new_cov = kernel_fn(X_new, X_new)
    assert new_cov.shape == (X_new.shape[0], X_new.shape[0])
