import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

import pytest
from gpax import RBFKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel, PolynomialKernel, GibbsKernel
from gpax.utils import constrain
from gpax.bijectors import Identity, Exp
from tests.utils import assert_same_pytree

from stheno import EQ, Matern12, Matern32, Matern52, Linear
import lab.jax as B



@pytest.mark.parametrize("kernel_fn", [RBFKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel, PolynomialKernel])
@pytest.mark.parametrize(
    "kernel_params", [{"lengthscale": 0.5, "variance": 0.7}, {"lengthscale": None, "variance": None}]
)
def test_initialize(kernel_fn, kernel_params):
    kernel = kernel_fn(**kernel_params)
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (10, 2))
    params = kernel.initialise_params(key, X)
    ls = kernel_params["lengthscale"] if kernel_params["lengthscale"] is not None else 1.0
    var = kernel_params["variance"] if kernel_params["variance"] is not None else 1.0
    params_expected = {"kernel": {"lengthscale": jnp.array([ls, ls]), "variance": jnp.array(var)}}
    print(params)
    print(params_expected)
    assert_same_pytree(params, params_expected)

    bijectors = kernel.get_bijectors()
    assert_same_pytree(bijectors, {"kernel": {"lengthscale": Exp(), "variance": Exp()}})


def test_combinations():
    X = jax.random.normal(jax.random.PRNGKey(0), (3, 1))
    kernel = (
        RBFKernel(lengthscale=0.1, variance=0.2) * Matern12Kernel(lengthscale=0.3, variance=0.4)
    ) * PolynomialKernel(lengthscale=1.0, variance=0.5)
    kernel_stheno = ((0.2 * EQ().stretch(0.1)) * (0.4 * Matern12().stretch(0.3))) * (0.5 + Linear())

    params = kernel.initialise_params(key=jax.random.PRNGKey(0), X=X)

    ours = kernel(params)(X, X)
    stheno_vals = kernel_stheno(X, X)

    assert jnp.allclose(ours, B.dense(stheno_vals))


def test_gibbs_kernel_dry_run():
    X_inducing = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    X = jax.random.normal(jax.random.PRNGKey(1), (6, 2))
    kernel = GibbsKernel(X_inducing=X_inducing)
    params = kernel.initialise_params(key=jax.random.PRNGKey(0), X_inducing=X_inducing)
    bijectrors = kernel.get_bijectors()

    params = constrain(params, bijectrors)


def test_gibbs_combinations():
    X_inducing = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    kernel = (RBFKernel() + GibbsKernel(X_inducing=X_inducing)) * PolynomialKernel()

    params = kernel.initialise_params(key=jax.random.PRNGKey(0), X_inducing=X_inducing)
    bijectrors = kernel.get_bijectors()

    params = constrain(params, bijectrors)
