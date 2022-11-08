import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

import pytest
from gpax.kernels import RBF, Matern12, Matern32, Matern52, Polynomial

# from gpax.special_kernels import Gibbs
from gpax.utils import constrain
from gpax.bijectors import Identity, Exp
from tests.utils import assert_same_pytree

import stheno
import lab.jax as B


@pytest.mark.parametrize(
    "X", [jax.random.normal(jax.random.PRNGKey(0), (3, 1)), jax.random.normal(jax.random.PRNGKey(0), (3, 2))]
)
@pytest.mark.parametrize("ls, var", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
@pytest.mark.parametrize(
    "kernel, stheno_kernel",
    [
        (RBF, stheno.EQ),
        (Matern12, stheno.Matern12),
        (Matern32, stheno.Matern32),
        (Matern52, stheno.Matern52),
        (Polynomial, stheno.Linear),
    ],
)
def test_execution(X, kernel, stheno_kernel, ls, var):
    if kernel is Polynomial:
        order = 1.0
        kernel = kernel(order=order)
        params = kernel.initialize_params(aux={"X": X})

        stheno_kernel = stheno_kernel().stretch(params["lengthscale"]) + params["variance"]

        ours = kernel(params)(X, X)
        stheno_vals = stheno_kernel(X, X)

        assert jnp.allclose(ours, B.dense(stheno_vals))

    else:
        kernel = kernel(lengthscale=ls, variance=var)
        params = kernel.initialize_params(aux={"X": X})
        assert len(params["lengthscale"]) == X.shape[1]
        kernel_stheno = params["variance"] * stheno_kernel().stretch(params["lengthscale"])
        ours = kernel(params)(X, X)
        stheno_vals = kernel_stheno(X, X)

        assert jnp.allclose(ours, B.dense(stheno_vals), atol=1e-2)


def test_combinations():
    X = jax.random.normal(jax.random.PRNGKey(0), (3, 1))
    kernel = (RBF(lengthscale=0.1, variance=0.2) * Matern12(lengthscale=0.3, variance=0.4)) * Polynomial(
        lengthscale=0.7, variance=0.5
    )
    kernel_stheno = ((0.2 * stheno.EQ().stretch(0.1)) * (0.4 * stheno.Matern12().stretch(0.3))) * (
        0.5 + stheno.Linear().stretch(0.7)
    )

    params = kernel.initialize_params(aux={"X": X})

    ours = kernel(params)(X, X)
    stheno_vals = kernel_stheno(X, X)

    assert jnp.allclose(ours, B.dense(stheno_vals))


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
