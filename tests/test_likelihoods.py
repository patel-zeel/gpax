import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

import pytest
import gpax.bijectors as gb

from gpax.models import ExactGPRegression
import gpax.kernels as gpk
import gpax.likelihoods as gpl
import gpax.means as gpm

# from gpax.likelihoods import HeteroscedasticGaussian
# from gpax.models import ExactGPRegression
# from gpax.means import Scalar
# from gpax.kernels import RBF

from tests.utils import assert_same_pytree


def test_gaussian():
    likelihood = gpl.Gaussian()
    params = likelihood.get_params(raw_dict=False)
    assert_same_pytree(params["scale"](), jnp.array(1.0))

    num = 0.3
    likelihood = gpl.Gaussian(scale=num)
    params = likelihood.get_params(raw_dict=False)
    assert_same_pytree(params["scale"](), jnp.array(num))

    likelihood.unconstrain()
    assert params["scale"]() == gb.get_positive_bijector().inverse(num)

    likelihood.constrain()
    assert params["scale"]() == jnp.array(num)

    likelihood.unconstrain()
    assert params["scale"]() == gb.get_positive_bijector().inverse(num)


def test_dynamic_default():
    likelihood = gpl.Gaussian()
    assert isinstance(likelihood.scale._bijector, gb.Exp)
    gb.set_positive_bijector(gb.Softplus)

    likelihood = gpl.Gaussian()
    assert isinstance(likelihood.scale._bijector, gb.Softplus)


def test_heteroscedastic_gaussian():
    num_datapoints = 10
    num_inducing = 3
    num_dims = 2

    X = jax.random.normal(jax.random.PRNGKey(0), (num_datapoints, num_dims))
    X_inducing = jax.random.normal(jax.random.PRNGKey(1), (num_inducing, num_dims))
    latent_gp = ExactGPRegression(kernel=gpk.RBF(input_dim=num_dims), mean=gpm.Scalar(), likelihood=gpl.Gaussian())
    likelihood = gpl.HeteroscedasticGaussian(latent_gp=latent_gp, X_inducing=X_inducing)

    likelihood.unconstrain()
    params = likelihood.get_params()
    assert params["scale"].shape == (num_inducing,)

    likelihood.constrain()
    infered_variance = likelihood(X)
    assert infered_variance.shape == (num_datapoints,)

    # One cycle
    likelihood.unconstrain()
    params = likelihood.get_params()
    assert params["scale"].shape == (num_inducing,)
