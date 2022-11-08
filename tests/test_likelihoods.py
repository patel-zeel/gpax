import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

import pytest
from gpax.likelihoods import Gaussian, HeteroscedasticGaussian
from gpax.models import ExactGPRegression
from gpax.means import Scalar
from gpax.kernels import RBF

from gpax.utils import constrain
from tests.utils import assert_same_pytree


def test_gaussian():
    likelihood = Gaussian()
    params = likelihood.initialize_params()
    assert_same_pytree(params, {"variance": jnp.array(1.0)})

    likelihood = Gaussian(variance=0.2)
    params = likelihood.initialize_params()
    assert_same_pytree(params, {"variance": jnp.array(0.2)})


def test_heteroscedastic_gaussian():
    num_datapoints = 10
    num_inducing = 3
    num_dims = 2

    X = jax.random.normal(jax.random.PRNGKey(0), (num_datapoints, num_dims))
    X_inducing = jax.random.normal(jax.random.PRNGKey(0), (num_inducing, 2))
    latent_gp = ExactGPRegression(kernel=RBF(), mean=Scalar(), likelihood=Gaussian())
    likelihood = HeteroscedasticGaussian(latent_gp=latent_gp)

    params = likelihood.initialize_params(key=jax.random.PRNGKey(0), aux={"X": X})
    assert params["whitened_raw_likelihood_variance"].shape == (num_datapoints,)
    params = constrain(params, likelihood.constraints)

    params = likelihood.initialize_params(key=jax.random.PRNGKey(0), aux={"X": X, "X_inducing": X_inducing})
    assert params["whitened_raw_likelihood_variance"].shape == (num_inducing,)
    params = constrain(params, likelihood.constraints)