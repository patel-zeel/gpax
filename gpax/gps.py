from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as tree_util

from gpax.kernels import Kernel, RBFKernel
from gpax.means import ConstantMean, Mean
from gpax.noises import HomoscedasticNoise, Noise
from gpax.bijectors import Identity

from typing import Literal, Union

from jaxtyping import Array


class AbstractGP:
    kernel: Kernel = RBFKernel()
    noise: Noise = HomoscedasticNoise()
    mean: Mean = ConstantMean()

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
            **self.mean.initialise_params(keys[0]),
            **kernels_params,
            **self.noise.initialise_params(keys[2], X_inducing=X_inducing),
        }
        key = jax.random.split(keys[3], 1)[0]
        return {**params, **self.__initialise_params__(key, X=X, X_inducing=X_inducing)}

    def __initialise_params__(self, key, X, X_inducing=None):
        return NotImplementedError("This method must be implemented by a subclass.")

    def get_bijectors(self):
        bijectors = {
            **self.mean.get_bijectors(),
            **self.kernel.get_bijectors(),
            **self.noise.get_bijectors(),
        }
        return {**bijectors, **self.__get_bijectors__()}

    def __get_bijectors__(self):
        return NotImplementedError("This method must be implemented by a subclass.")


class ExactGP(AbstractGP):
    def add_noise(self, K, noise):
        rows, columns = jnp.diag_indices_from(K)
        return K.at[rows, columns].set(K[rows, columns] + noise + jnp.jitter)

    def log_probability(self, params, X, y, prior):
        noise = self.noise(params, X)
        mean = self.mean(params)
        kernel = self.kernel(params)
        covariance = kernel(X, X)
        noisy_covariance = self.add_noise(covariance, noise)
        chol = jnp.linalg.cholesky(noisy_covariance)
        y_bar = y - mean
        k_inv_y = jsp.linalg.solve_triangular(chol, y_bar, lower=True)
        log_likelihood = (
            -0.5 * (y_bar.ravel() * k_inv_y.ravel()).sum()  # This will break for multi-dimensional y
            - jnp.log(chol.diagonal()).sum()
            - 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)
        )
        log_prior = tree_util.tree_map(lambda _prior, param: _prior.log_prob(param).sum(), prior, params)

        return log_likelihood + log_prior

    def predict(self, params, X, y, X_test, return_cov=True, include_noise=True):
        mean = self.mean(params)
        kernel = self.kernel(params)
        y_bar = y - mean
        noisy_covariance = self.add_noise(kernel(X, X), self.noise(params, X) + jnp.jitter)
        L = jnp.linalg.cholesky(noisy_covariance)
        alpha = jsp.linalg.cho_solve((L, True), y_bar)
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

    def __initialise_params__(self, key, X, X_inducing):
        return {}  # No additional parameters to initialise.

    def __get_bijectors__(self):
        return {}  # No additional bijectors to return.


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
