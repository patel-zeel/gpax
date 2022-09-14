import jax
import jax.numpy as jnp

import pytest
from gpax import HomoscedasticNoise, HeteroscedasticNoise
from gpax.utils import constrain
from tests.utils import assert_same_pytree

import lab.jax as B

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors


def test_initialize():
    noise = HomoscedasticNoise()
    params = noise.initialise_params(key=jax.random.PRNGKey(0))
    assert_same_pytree(params, {"noise": {"variance": jnp.array(1.0)}})

    noise = HomoscedasticNoise(variance=0.2)
    params = noise.initialise_params(key=jax.random.PRNGKey(0))
    assert_same_pytree(params, {"noise": {"variance": jnp.array(0.2)}})


def test_heteroscedastic_noise_dry_run():
    X_inducing = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    noise = HeteroscedasticNoise(X_inducing=X_inducing)

    params = noise.initialise_params(key=jax.random.PRNGKey(0))
    bijectors = noise.get_bijectors()

    params = constrain(params, bijectors)
