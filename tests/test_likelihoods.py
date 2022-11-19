import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

import pytest
import gpax.bijectors as gb
import gpax.distributions as gd

from gpax.models import ExactGPRegression
from gpax.core import Parameter
import gpax.kernels as gpk
import gpax.likelihoods as gpl
import gpax.means as gpm

# from gpax.likelihoods import HeteroscedasticGaussian
# from gpax.models import ExactGPRegression
# from gpax.means import Scalar
# from gpax.kernels import RBF

from tests.utils import assert_same_pytree, assert_approx_same_pytree


def test_gaussian():
    likelihood = gpl.Gaussian()
    params = likelihood.get_params(raw_dict=False)
    assert_same_pytree(params["scale"](), jnp.array(1.0))

    num = 0.3
    likelihood = gpl.Gaussian(scale=num)
    params = likelihood.get_params(raw_dict=False)
    assert_approx_same_pytree(params["scale"](), jnp.array(num))


def test_dynamic_default():
    likelihood = gpl.Gaussian()
    assert isinstance(likelihood.scale._bijector, type(gb.get_positive_bijector()))
    gb.set_positive_bijector(gb.Softplus)

    likelihood = gpl.Gaussian()
    assert isinstance(likelihood.scale._bijector, gb.Softplus)


def test_heteroscedastic_gaussian():
    num_datapoints = 10
    num_inducing = 3
    num_dims = 2

    X = jax.random.normal(jax.random.PRNGKey(0), (num_datapoints, num_dims))
    X_inducing = jax.random.normal(jax.random.PRNGKey(1), (num_inducing, num_dims))
    X_inducing = Parameter(X_inducing, fixed_init=True)
    latent_gp = ExactGPRegression(kernel=gpk.RBF(input_dim=num_dims), mean=gpm.Scalar(), likelihood=gpl.Gaussian())
    bijector = gb.InverseWhite(latent_gp=latent_gp, X_inducing=X_inducing)
    prior = bijector(gd.Normal(loc=0.0, scale=1.0))
    scale_inducing = Parameter(1.0, bijector, prior)
    likelihood = gpl.HeteroscedasticGaussian(scale_inducing=scale_inducing)

    params = likelihood.get_params()
    assert params["scale_inducing"].shape == (num_inducing,)

    infered_variance = likelihood(X)
    assert infered_variance.shape == (num_datapoints,)
