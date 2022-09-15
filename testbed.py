import jax
import jax.numpy as jnp

import optax

import matplotlib.pyplot as plt

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from gpax import ExactGP, SparseGP, RBFKernel, HomoscedasticNoise, ConstantMean
from gpax.utils import constrain, unconstrain, randomize, train_fn
from gpax.plotting import plot_posterior

import lab.jax as B

import pprint

pp = pprint.PrettyPrinter(depth=4)

X = jnp.linspace(-1, 1, 100).reshape(-1, 1)
# X = jnp.concatenate([X[:40], X[60:]])
key = jax.random.PRNGKey(0)
y = jnp.sin(2 * jnp.pi * X) + jax.random.normal(key, X.shape) * 0.2

plt.scatter(X, y)

## Exact GP

model = ExactGP(kernel=RBFKernel(), noise=HomoscedasticNoise(), mean=ConstantMean())


def loss_fun(params):
    return -model.log_probability(params, X, y)


key = jax.random.PRNGKey(123)
params = model.initialise_params(key, X)
pp.pprint(params)

bijectors = model.get_bijectors()
pp.pprint(bijectors)

params = model.initialise_params(key, X)
jax.grad(loss_fun)(params)
