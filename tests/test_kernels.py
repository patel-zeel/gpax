import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.tree_util as jtu
import jax.numpy as jnp

import pytest
import gpax.kernels as gpk
from gpax.models import ExactGPRegression
from gpax.core import Parameter
import gpax.kernels as gpk

# from gpax.special_kernels import Gibbs
from tests.utils import assert_same_pytree

# import stheno
# import lab.jax as B


# @pytest.mark.parametrize(
#     "X", [jax.random.normal(jax.random.PRNGKey(0), (3, 1)), jax.random.normal(jax.random.PRNGKey(0), (3, 2))]
# )
# @pytest.mark.parametrize("ls, scale", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
# @pytest.mark.parametrize(
#     "kernel, stheno_kernel",
#     [
#         (gpk.RBF, stheno.EQ),
#         (gpk.Matern12, stheno.Matern12),
#         (gpk.Matern32, stheno.Matern32),
#         (gpk.Matern52, stheno.Matern52),
#         (gpk.Polynomial, stheno.Linear),
#     ],
# )
# def test_execution(X, kernel, stheno_kernel, ls, scale):
#     if kernel is gpk.Polynomial:
#         order = 1.0
#         kernel = kernel(X=X, order=order)
#         kernel_fn = kernel.eval().get_kernel_fn()
#         params = kernel.get_parameters()

#         stheno_kernel = stheno_kernel() + params["center"]

#         ours = kernel_fn(X, X)
#         stheno_vals = stheno_kernel(X, X)

#         assert jnp.allclose(ours, B.dense(stheno_vals))

#     else:
#         kernel = kernel(X=X, lengthscale=ls, scale=scale, ARD=True)
#         kernel_fn = kernel.eval().get_kernel_fn()
#         params = kernel.get_parameters()
#         assert len(params["lengthscale"]) == X.shape[1]
#         kernel_stheno = (params["scale"] ** 2) * stheno_kernel().stretch(params["lengthscale"])
#         ours = kernel_fn(X, X)
#         stheno_vals = kernel_stheno(X, X)

#         assert jnp.allclose(ours, B.dense(stheno_vals), atol=1e-2)


# def test_combinations():
#     X = jax.random.normal(jax.random.PRNGKey(0), (3, 1))
#     kernel = (
#         gpk.RBF(X=X, lengthscale=0.1, scale=0.2) * gpk.Matern12(X=X, lengthscale=0.3, scale=0.4)
#     ) * gpk.Polynomial(X=X, center=0.5, order=1.0)
#     kernel_stheno = ((0.2**2 * stheno.EQ().stretch(0.1)) * (0.4**2 * stheno.Matern12().stretch(0.3))) * (
#         0.5 + stheno.Linear()
#     )

#     ours = kernel.eval().get_kernel_fn()(X, X)
#     stheno_vals = kernel_stheno(X, X)

#     assert jnp.allclose(ours, B.dense(stheno_vals), atol=1e-2)


@pytest.mark.parametrize("kernel", [gpk.GibbsDeltaInducing, gpk.GibbsHeinonen])
def test_gibbs_kernel(kernel):
    X_inducing = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    X = jax.random.normal(jax.random.PRNGKey(0), (6, 2))
    X_new = jax.random.normal(jax.random.PRNGKey(0), (12, 2))

    if kernel is gpk.GibbsHeinonen:
        X_inducing = X

    kernel = kernel(
        flex_lengthscale=True,
        flex_scale=True,
        latent_lengthscale_ell=0.1,
        latent_scale_ell=0.2,
        latent_lengthscale_sigma=0.3,
        latent_scale_sigma=0.4,
        X_inducing=X_inducing,
    )

    kernel_fn = kernel.get_kernel_fn(Parameter(X_inducing))
    cov, log_prior = kernel_fn(X, X)
    assert cov.shape == (X.shape[0], X.shape[0])

    kernel_fn = kernel.eval().get_kernel_fn(Parameter(X_inducing))
    cross_cov = kernel_fn(X, X_new)
    assert cross_cov.shape == (X.shape[0], X_new.shape[0])

    new_cov = kernel_fn(X_new, X_new)
    assert new_cov.shape == (X_new.shape[0], X_new.shape[0])
