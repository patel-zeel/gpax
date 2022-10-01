from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
import jax.tree_util as tree_util

from gpax.kernels import Kernel, RBFKernel
from gpax.means import ConstantMean, Mean
from gpax.noises import HomoscedasticNoise, Noise
from gpax.bijectors import Identity

from typing import Literal, Union

from jaxtyping import Array

from gpax.utils import get_raw_log_prior


class AbstractGP:
    def __init__(self, kernel=RBFKernel(), noise=HomoscedasticNoise(), mean=ConstantMean()):
        self.kernel = kernel
        self.noise = noise
        self.mean = mean

    @abstractmethod
    def log_probability(self, params, X, y):
        return NotImplementedError("This method must be implemented by a subclass.")

    @abstractmethod
    def predict(self, params, X, X_new, return_cov=True, include_noise=True):
        return NotImplementedError("This method must be implemented by a subclass.")

    def initialise_params(self, key, X, X_inducing=None):
        keys = jax.random.split(key, 4)
        if self.kernel.__class__.__name__ == "GibbsKernel":
            kernels_params = self.kernel.initialise_params(keys[0], X_inducing=X_inducing)
        elif self.kernel.__class__.__name__ in ["SumKernel", "ProductKernel"]:
            kernels_params = self.kernel.initialise_params(keys[0], X=X, X_inducing=X_inducing)
        else:
            kernels_params = self.kernel.initialise_params(keys[0], X=X)

        params = {
            **self.mean.initialise_params(keys[1]),
            **kernels_params,
            **self.noise.initialise_params(keys[2], X_inducing=X_inducing),
        }
        return {**params, **self.__initialise_params__(keys[3], X=X, X_inducing=X_inducing)}

    def __initialise_params__(self, key, X, X_inducing=None):
        return NotImplementedError("This method must be implemented by a subclass.")

    def get_bijectors(self):
        bijectors = {
            **self.mean.get_bijectors(),
            **self.kernel.get_bijectors(),
            **self.noise.get_bijectors(),
        }
        return {**bijectors, **self.__get_bijectors__()}

    def get_priors(self):
        priors = {
            **self.mean.get_priors(),
            **self.kernel.get_priors(),
            **self.noise.get_priors(),
        }
        return {**priors, **self.__get_priors__()}

    def __get_bijectors__(self):
        return NotImplementedError("This method must be implemented by a subclass.")


class ExactGP(AbstractGP):
    def add_noise(self, K, noise):
        rows, columns = jnp.diag_indices_from(K)
        return K.at[rows, columns].set(K[rows, columns] + noise + jnp.jitter)

    def log_probability(self, params, X, y):
        noise = self.noise(params, X)
        mean = self.mean(params)
        kernel = self.kernel(params)
        covariance = kernel(X, X)
        noisy_covariance = self.add_noise(covariance, noise)
        chol = jnp.linalg.cholesky(noisy_covariance)
        y_bar = y - mean
        k_inv_y = jsp.linalg.cho_solve((chol, True), y_bar)
        log_likelihood = (
            -0.5 * (y_bar.ravel() * k_inv_y.ravel()).sum()  # This will break for multi-dimensional y
            - jnp.log(chol.diagonal()).sum()
            - 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)
        )
        # print(
        #     -(y_bar.ravel() * k_inv_y.ravel()).sum(),
        #     -2 * jnp.log(chol.diagonal()).sum(),
        #     -y.shape[0] * jnp.log(2 * jnp.pi),
        # )
        return log_likelihood

    def log_prior(self, params):
        prior = self.get_priors()
        bijectors = self.get_bijectors()
        log_prior = get_raw_log_prior(prior, params, bijectors)
        return ravel_pytree(log_prior)[0].sum()

    def condition(self, params, X, y):
        mean = self.mean(params)
        kernel = self.kernel(params)
        y_bar = y - mean
        noisy_covariance = self.add_noise(kernel(X, X), self.noise(params, X) + jnp.jitter)
        L = jnp.linalg.cholesky(noisy_covariance)
        alpha = jsp.linalg.cho_solve((L, True), y_bar)

        def predict_fn(X_test, return_cov=True, include_noise=True):
            K_star = kernel(X_test, X)
            pred_mean = K_star @ alpha + mean

            if return_cov:
                v = jsp.linalg.cho_solve((L, True), K_star.T)
                pred_cov = kernel(X_test, X_test) - K_star @ v
                if include_noise:
                    pred_cov = self.add_noise(pred_cov, self.noise(params, X_test))
                return pred_mean, pred_cov
            else:
                return pred_mean

        return predict_fn

    def predict(self, params, X, y, X_test, return_cov=True, include_noise=True):
        predict_fn = self.condition(params, X, y)
        return predict_fn(X_test, return_cov=return_cov, include_noise=include_noise)

    def __initialise_params__(self, key, X, X_inducing):
        return {}  # No additional parameters to initialise.

    def __get_bijectors__(self):
        return {}  # No additional bijectors to return.

    def __get_priors__(self):
        return {}  # No additional priors to return.


class SparseGP(AbstractGP):
    method: Literal["vfe", "fitc", "dtc"] = "vfe"
    X_inducing: Array = None

    def log_probability(self, params, X, y):
        X_inducing = params["X_inducing"]
        k_mm = self.kernel(params, X_inducing, X_inducing)

    def predict(self, params, X, y, X_test):
        pass

    def __initialise_params__(self, key, X, X_inducing):
        if X_inducing is None:
            assert self.X_inducing is not None, "X_inducing must be specified."
            X_inducing = self.X_inducing
        return {"X_inducing": X_inducing}

    def __get_bijectors__(self):
        return {"X_inducing": Identity()}
