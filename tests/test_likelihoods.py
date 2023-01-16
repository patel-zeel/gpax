import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

import pytest

from gpax.models import ExactGPRegression, LatentGPHeinonen, LatentGPDeltaInducing
from gpax.core import Parameter, get_positive_bijector
from gpax.kernels import RBF, Scale
from gpax.likelihoods import Gaussian, Heteroscedastic

from tests.utils import assert_same_pytree, assert_approx_same_pytree

X_inducing = jax.random.uniform(jax.random.PRNGKey(0), (5, 3))
X = jax.random.uniform(jax.random.PRNGKey(0), (10, 3))
X_new = jax.random.uniform(jax.random.PRNGKey(1), (15, 3))


def test_gaussian():
    likelihood = Gaussian()
    assert likelihood.scale() == 1.0
    assert isinstance(likelihood.scale.bijector, get_positive_bijector().__class__)
    params = likelihood.get_parameters()
    assert_same_pytree(params, {"scale": 1.0})

    likelihood_fn = likelihood.get_likelihood_fn()
    scale, log_prior = likelihood_fn(X)
    assert scale == 1.0


@pytest.mark.parametrize("X_inducing", [X_inducing])
@pytest.mark.parametrize("latent_model_type", [LatentGPHeinonen, LatentGPDeltaInducing])
def test_heteroscedastic(latent_model_type, X_inducing):
    X_inducing = X_inducing if latent_model_type is LatentGPDeltaInducing else X
    base_kernel = RBF(X_inducing, lengthscale=2.0)
    kernel = Scale(X, base_kernel, variance=3.0**2)
    latent_model = latent_model_type(X_inducing, kernel)
    likelihood = Heteroscedastic(latent_model)
    params = likelihood.get_parameters()
    assert_same_pytree(
        params,
        {
            "latent_model": {
                "latent": jnp.ones(()).repeat(X_inducing.shape[0]),
                "kernel": {
                    "variance": jnp.array(3.0**2),
                    "base_kernel": {
                        "lengthscale": jnp.array(2.0).repeat(X.shape[1]),
                    },
                },
            }
        },
    )

    likelihood_fn = likelihood.get_likelihood_fn(X_inducing)
    scale, log_prior = likelihood_fn(X)
    assert scale.shape == (X.shape[0],)
    likelihood.eval()
    scale = likelihood_fn(X)
    scale_new = likelihood_fn(X_new)
    assert scale.shape == (X.shape[0],)
    assert scale_new.shape == (X_new.shape[0],)
