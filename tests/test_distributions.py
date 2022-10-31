import jax
import jax.numpy as jnp

# jax enable x64
jax.config.update("jax_enable_x64", True)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from gpax.distributions import Normal, Beta, Exponential, Gamma
from gpax.bijectors import Log, Exp, Sigmoid

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


normal1 = (Normal, tfd.Normal, {"loc": 0.3, "scale": 1.2})
normal2 = (Normal, tfd.Normal, {"loc": 1.6, "scale": 2.9})
beta1 = (Beta, tfd.Beta, {"concentration0": 0.3, "concentration1": 1.2})
beta2 = (Beta, tfd.Beta, {"concentration0": 1.6, "concentration1": 2.9})
gamma1 = (Gamma, tfd.Gamma, {"concentration": 0.3, "rate": 1.2})
gamma2 = (Gamma, tfd.Gamma, {"concentration": 1.6, "rate": 2.9})
exp1 = (Exponential, tfd.Exponential, {"rate": 0.3})
exp2 = (Exponential, tfd.Exponential, {"rate": 1.6})
n_samples = 100


@pytest.mark.parametrize("our_dist, tfd_dist, params", [normal1, normal2, beta1, beta2, gamma1, gamma2, exp1, exp2])
def test_distribution(our_dist, tfd_dist, params):
    our_dist = our_dist(**params)
    tfd_dist = tfd_dist(**params)

    samples = our_dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(n_samples,))
    assert jnp.allclose(our_dist.log_prob(samples), tfd_dist.log_prob(samples), atol=1e-6)

    samples = tfd_dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(n_samples,))
    assert jnp.allclose(our_dist.log_prob(samples), tfd_dist.log_prob(samples), atol=1e-6)


@pytest.mark.parametrize("our_dist, tfd_dist, params", [normal1, normal2, beta1, beta2, gamma1, gamma2, exp1, exp2])
@pytest.mark.parametrize("our_bijector, tfb_bijector", [(Exp(), tfb.Exp()), (Sigmoid(), tfb.Sigmoid())])
def test_transformed_distribution(our_dist, tfd_dist, params, our_bijector, tfb_bijector):
    our_dist = our_bijector(our_dist(**params))
    tfd_dist = tfb_bijector(tfd_dist(**params))

    samples = our_dist.sample(seed=jax.random.PRNGKey(1), sample_shape=(n_samples,))
    assert jnp.allclose(
        our_dist.log_prob(samples),
        tfd_dist.log_prob(samples),
        atol=1e-1,
    )

    samples = tfd_dist.sample(seed=jax.random.PRNGKey(2), sample_shape=(n_samples,))
    our_log_prob = our_dist.log_prob(samples)
    tfb_log_prob = tfd_dist.log_prob(samples)
    assert jnp.allclose(our_log_prob, tfb_log_prob, atol=1e-1)
