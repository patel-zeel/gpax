import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

import optax

import matplotlib.pyplot as plt

from gpax import ExactGP, SparseGP, GibbsKernel, HomoscedasticNoise, HeteroscedasticNoise
from gpax.utils import constrain, unconstrain, randomize, train_fn
from gpax.plotting import plot_posterior

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors

from stheno import GP, EQ

import lab.jax as B
from matrix import Dense, dense

import regdata as rd

import pprint

pp = pprint.PrettyPrinter(depth=4)

from jax.config import config

config.update("jax_debug_nans", True)

# X = jnp.linspace(-1, 1, 100).reshape(-1, 1)
# # X = jnp.concatenate([X[:40], X[60:]])
# key = jax.random.PRNGKey(0)
# y = jnp.sin(2 * jnp.pi * X) + jax.random.normal(key, X.shape)*0.2

X, y, X_test = rd.MotorcycleHelmet().get_data()

X_inducing = X[::10]

kernel = GibbsKernel(X_inducing=X_inducing)
noise = HeteroscedasticNoise(use_kernel_inducing=True)
model = ExactGP(kernel=kernel, noise=noise)


def loss_fun(params):
    return -model.log_probability(params, X, y)


key = jax.random.PRNGKey(0)
params = model.initialise_params(key, X, X_inducing)
bijectors = model.get_bijectors()

params = unconstrain(params, bijectors)
key = jax.random.PRNGKey(1)
params = randomize(params, key)
loss_fun(constrain(params, bijectors))
