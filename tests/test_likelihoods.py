import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

import pytest

from gpax.models import ExactGPRegression
from gpax.core import Parameter, get_positive_bijector
from gpax.kernels import RBF
from gpax.likelihoods import Gaussian, HeteroscedasticHeinonen, HeteroscedasticDeltaInducing

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


@pytest.mark.parametrize("likelihood_type", [HeteroscedasticHeinonen, HeteroscedasticDeltaInducing])
def test_heteroscedastic(likelihood_type):
    X_inducing_tmp = X_inducing if likelihood_type is HeteroscedasticDeltaInducing else X
    likelihood = likelihood_type(X_inducing_tmp, 2.0, 3.0, RBF)
    params = likelihood.get_parameters()
    assert_same_pytree(
        params,
        {
            "latent_gp": {
                "latent": jnp.zeros(()).repeat(X_inducing_tmp.shape[0]),
                "lengthscale": jnp.array(2.0).repeat(X.shape[1]),
                "scale": 3.0,
            }
        },
    )

    X_inducing_p = Parameter(X_inducing_tmp, fixed_init=True)
    likelihood_fn = likelihood.get_likelihood_fn(X_inducing_p)
    scale, log_prior = likelihood_fn(X)
    assert scale.shape == (X.shape[0],)
    likelihood.eval()
    scale, scale_new = likelihood_fn(X, X_new)
    assert scale.shape == (X.shape[0],)
    assert scale_new.shape == (X_new.shape[0],)
