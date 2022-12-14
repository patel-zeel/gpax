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


@pytest.mark.parametrize("method", ["gp_neurips", "heinonen"])
def test_heteroscedastic_gp_neurips(method):
    num_datapoints = 10
    num_test = 15
    num_inducing = 3
    num_dims = 2

    X = jax.random.normal(jax.random.PRNGKey(0), (num_datapoints, num_dims))
    X_new = jax.random.normal(jax.random.PRNGKey(1), (num_test, num_dims))

    if method == "gp_neurips":
        X_inducing = jax.random.normal(jax.random.PRNGKey(1), (num_inducing, num_dims))
    elif method == "heinonen":
        X_inducing = X
    else:
        raise ValueError(f"Unknown method {method}")
    X_inducing = Parameter(X_inducing, fixed_init=True)
    latent_gp = ExactGPRegression(kernel=gpk.RBF(input_dim=num_dims), mean=gpm.Scalar(), likelihood=gpl.Gaussian())
    bijector = gb.InverseWhite(latent_gp=latent_gp, X_inducing=X_inducing)
    prior = bijector(gd.Normal(loc=0.0, scale=1.0))
    inversed_prior = gb.get_positive_bijector()(gd.Normal(loc=0.0, scale=1.0))
    scale_inducing = Parameter(1.0, bijector, prior, inversed_init=True, inversed_prior=inversed_prior)
    likelihood = gpl.HeteroscedasticGaussian(scale_inducing=scale_inducing)

    params = likelihood.get_params()
    if method == "gp_neurips":
        assert params["scale_inducing"].shape == (num_inducing,)
    elif method == "heinonen":
        assert params["scale_inducing"].shape == (num_datapoints,)
    else:
        raise ValueError(f"Unknown method {method}")

    infered_variance = likelihood(X_new, train_mode=False)
    assert infered_variance.shape == (num_test,)
