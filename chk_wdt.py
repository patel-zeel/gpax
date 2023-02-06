import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

import matplotlib.pyplot as plt
import arviz as az

import gpax.kernels as gpk
import gpax.likelihoods as gpl
import gpax.models as gpm
from gpax.means import Scalar, Average


X = jnp.linspace(0, 1, 20).reshape(-1, 1)
lengthscale = 0.1
variance = 0.5

base_kernel = gpk.RBF(X, lengthscale=0.1)
kernel = variance * base_kernel
print(kernel)

base_kernel = gpk.RBF(X, lengthscale=lengthscale)
kernel2 = gpk.Scale(X, base_kernel, variance=variance)
print(kernel2)
kernel_fn = kernel.get_kernel_fn()
K, log_prior = kernel_fn(X, X)
print(K.shape, log_prior)

lengthscale = 0.1

matern32 = variance * gpk.Matern32(X, lengthscale)
matern12 = variance * gpk.Matern12(X, lengthscale)
matern52 = variance * gpk.Matern52(X, lengthscale)
periodic = variance * gpk.Periodic(X, lengthscale, period=0.2)

latent_kernel = variance * gpk.RBF(X, lengthscale=0.1)
latent_model = gpm.LatentGPHeinonen(X, latent_kernel)

base_kernel = gpk.RBF(X, lengthscale=0.2)
kernel = gpk.InputDependentScale(X, base_kernel, latent_model)
print(kernel)
# Let us plot the input-dependent variance ($\sigma^2$).
def plot_kernel():
    variable_scale, _ = kernel.latent_model(X_inducing=X)(X=X)
    variable_variance = variable_scale**2

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(X, variable_variance, label="constant variance")

    K, log_prior = kernel.get_kernel_fn(X_inducing=X)(X, X)

    ax[1].imshow(K)
    return ax


# Fixing a constant variance for all inputs. This could be a good initialization for the input-dependent variance kernel.
kernel.latent_model.reverse_init(0.1)
ax = plot_kernel()
ax[0].set_ylim(0, 0.02)
# Randomizing the variance for all inputs:
key = jax.random.PRNGKey(10)
kernel.initialize(key)
plot_kernel()
## Models
X = jnp.linspace(0, 1, 50).reshape(-1, 1)
X_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
key = jax.random.PRNGKey(10)
y = (jnp.sin(2 * jnp.pi * X) + jax.random.normal(key, X.shape) * 0.2).squeeze()

plt.scatter(X, y)
kernel = 1.0 * gpk.RBF(X, lengthscale=0.1)
model = gpm.ExactGPRegression(kernel, gpl.Gaussian(), Scalar())
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 5)


def fit_fn(key):
    # model.fit(key, X, y, lr=0.01, epochs=100)
    return model


models = jax.vmap(fit_fn)(keys)

# pred_mean, pred_cov = model.predict(X, y, X_test)

# plt.scatter(X, y)
# plt.plot(X_test, pred_mean, color='red');
# plt.fill_between(
#     X_test.squeeze(),
#     pred_mean - 2 * jnp.sqrt(pred_cov.diagonal()),
#     pred_mean + 2 * jnp.sqrt(pred_cov.diagonal()),
#     color='red',
#     alpha=0.2,
# );
# (latent-models)=
### Latent Models
