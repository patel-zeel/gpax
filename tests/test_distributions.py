import jax
import jax.numpy as jnp

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from gpax.distributions import Normal, Beta
from gpax.bijectors import Log, Exp, Sigmoid

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


@pytest.mark.parametrize("loc", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("bijector, tfb_bijector", [(Exp(), tfb.Exp())])
def test_normal(loc, scale, bijector, tfb_bijector):
    scalar = jax.random.normal(jax.random.PRNGKey(0))
    tensor = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 5))

    normal = Normal(loc=loc, scale=scale)
    tfd_normal = tfd.Normal(loc=loc, scale=scale)

    assert jnp.allclose(normal.log_prob(scalar), tfd_normal.log_prob(scalar))
    assert jnp.allclose(normal.log_prob(tensor), tfd_normal.log_prob(tensor))

    transformed_normal = bijector(normal)
    tfd_transformed_normal = tfb_bijector(tfd_normal)

    assert jnp.allclose(
        transformed_normal.log_prob(bijector(scalar)), tfd_transformed_normal.log_prob(tfb_bijector(scalar))
    )
    assert jnp.allclose(
        transformed_normal.log_prob(bijector(tensor)), tfd_transformed_normal.log_prob(tfb_bijector(tensor))
    )


@pytest.mark.parametrize("concentration0", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("concentration1", [0.1, 1.2, 2.0])
@pytest.mark.parametrize("bijector, tfb_bijector", [(Log(), tfb.Log()), (Sigmoid(), tfb.Sigmoid())])
def test_beta(concentration0, concentration1, bijector, tfb_bijector):
    scalar = jax.random.uniform(jax.random.PRNGKey(0))
    tensor = jax.random.uniform(jax.random.PRNGKey(1), (2, 3, 5))

    beta = Beta(concentration0=concentration0, concentration1=concentration1)
    tfd_beta = tfd.Beta(concentration0=concentration0, concentration1=concentration1)

    assert jnp.allclose(beta.log_prob(scalar), tfd_beta.log_prob(scalar))
    assert jnp.allclose(beta.log_prob(tensor), tfd_beta.log_prob(tensor))

    transformed_beta = bijector(beta)
    tfd_transformed_beta = tfb_bijector(tfd_beta)

    assert jnp.allclose(
        transformed_beta.log_prob(bijector(scalar)), tfd_transformed_beta.log_prob(tfb_bijector(scalar))
    )
    assert jnp.allclose(
        transformed_beta.log_prob(bijector(tensor)),
        tfd_transformed_beta.log_prob(tfb_bijector(tensor)),
        rtol=1e-4,
        atol=1e-4,
    )
