from __future__ import annotations
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from gpax.core import Module, Parameter
from gpax.defaults import get_default_jitter
import gpax.distributions as gd
import gpax.bijectors as gb
from gpax.utils import add_to_diagonal, get_a_inv_b, train_fn

from jaxtyping import Array
from typing import TYPE_CHECKING

from chex import dataclass
from copy import deepcopy

if TYPE_CHECKING:
    from gpax.kernels import Kernel
    from gpax.likelihoods import Likelihood
    from gpax.means import Mean


class Model(Module):
    pass


@dataclass
class ExactGPRegression(Model):
    kernel: Kernel = None
    likelihood: Likelihood = None
    mean: Mean = None
    X_inducing: Array = None

    def __post_init__(self):
        assert self.kernel is not None, "kernel must be provided"
        assert self.likelihood is not None, "likelihood must be provided"
        assert self.mean is not None, "mean must be provided"
        if self.X_inducing is not None:
            X_inducing = Parameter(self.X_inducing, prior=gd.Fixed())
            self.common_params = {"X_inducing": X_inducing}
            self.kernel.common_params = self.common_params
            self.likelihood.common_params = self.common_params

    def __get_params__(self):
        params = {
            "kernel": self.kernel.__get_params__(),
            "likelihood": self.likelihood.__get_params__(),
            "mean": self.mean.__get_params__(),
        }
        if self.X_inducing is not None:
            params["X_inducing"] = self.common_params["X_inducing"]
        return params

    def set_params(self, params):
        self.kernel.set_params(params["kernel"])
        self.likelihood.set_params(params["likelihood"])
        self.mean.set_params(params["mean"])
        if self.X_inducing is not None:
            self.common_params["X_inducing"].set(params["X_inducing"])

    def log_probability(self, X, y):
        """
        prior_type: default: None, possible values: "prior", "posterior", None
        """
        covariance = self.kernel(X, X)
        noise_scale = self.likelihood(X)

        y_bar = y - self.mean(y=y)
        noisy_covariance = add_to_diagonal(covariance, noise_scale**2, get_default_jitter())

        k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)

        log_likelihood = (
            -0.5 * (y_bar.ravel() * k_inv_y.ravel()).sum()  # This will break for multi-dimensional y
            - jnp.log(k_cholesky.diagonal()).sum()
            - 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)
        )

        return log_likelihood

    def optimize(self, key, optimizer, X, y, n_iters, include_prior=True, lax_scan=True):
        self.initialize(key)
        self.unconstrain()
        raw_params = self.get_params()

        def loss_fn(raw_params):
            model = deepcopy(self)
            model.unconstrain()
            model.set_params(raw_params)

            # Prior should be computed before constraining the parameters
            log_prior = 0.0
            if include_prior:
                log_prior = model.log_prior()

            model.constrain()
            log_prob = model.log_probability(X, y)
            return -log_prob - log_prior

        result = train_fn(loss_fn, raw_params, optimizer, n_iters, lax_scan=lax_scan)
        self.unconstrain()
        self.set_params(result["raw_params"])
        self.constrain()
        return result

    def condition(self, X, y):
        """
        This function is useful while doing batch prediction.
        """
        train_cov = self.kernel(X, X)
        noise_scale = self.likelihood(X)

        mean = self.mean(y=y)
        y_bar = y - mean
        noisy_covariance = add_to_diagonal(train_cov, noise_scale**2, get_default_jitter())
        k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)

        def predict_fn(X_test, return_cov=True, include_noise=True):
            K_star = self.kernel(X_test, X)
            pred_mean = K_star @ k_inv_y + mean

            if return_cov:
                k_inv_k_star = jsp.linalg.cho_solve((k_cholesky, True), K_star.T)
                pred_cov = self.kernel(X_test, X_test) - (K_star @ k_inv_k_star)
                if include_noise:
                    pred_cov = add_to_diagonal(pred_cov, self.likelihood(X_test) ** 2, get_default_jitter())
                return pred_mean, pred_cov
            else:
                return pred_mean

        return predict_fn

    def predict(self, X, y, X_test, return_cov=True, include_noise=True):
        """
        This method is suitable for one time prediction.
        In case of batch prediction, it is better to use `condition` method in combination with `predict`.
        """
        predict_fn = self.condition(X, y)
        return predict_fn(X_test, return_cov=return_cov, include_noise=include_noise)


# class SparseGPRegression(Model):
#     def __init__(self, X_inducing, kernel, noise, mean):
#         super().__init__(kernel, noise, mean)
#         self.X_inducing = X_inducing

#     def log_probability(self, params, X, y, include_prior=True):
#         X_inducing = params["X_inducing"]
#         kernel_fn = self.kernel(params)
#         prior_log_prob = 0.0
#         if self.noise.__class__.__name__ == "HeinonenHeteroscedasticNoise":
#             if include_prior:
#                 _, tmp_prior_log_prob = self.noise.train_noise(params, return_prior_log_prob=True)
#                 prior_log_prob += tmp_prior_log_prob

#         if self.mean.__class__.__name__ == "ZeroMean":
#             mean = y.mean()
#         else:
#             mean = self.mean(params)

#         if self.kernel.__class__.__name__ == "HeinonenGibbsKernel":
#             if include_prior:
#                 k_mm, tmp_prior_log_prob = self.kernel.train_cov(params, return_prior_log_prob=True)
#                 prior_log_prob += tmp_prior_log_prob
#             else:
#                 k_mm = self.kernel.train_cov(params, return_prior_log_prob=False)
#         else:
#             k_mm = kernel_fn(X_inducing, X_inducing)

#         y_bar = y - mean
#         noise_n = self.noise(params, X).squeeze()
#         k_mm = k_mm + jnp.eye(X_inducing.shape[0]) * jnp.jitter
#         k_nm = kernel_fn(X, X_inducing)

#         # woodbury identity
#         left = k_nm / noise_n.reshape(-1, 1)
#         right = left.T
#         middle = k_mm + right @ k_nm
#         k_inv = jnp.diag(1 / noise_n.squeeze()) - left @ jsp.linalg.cho_solve(
#             (jnp.linalg.cholesky(middle), True), right
#         )
#         data_fit = y_bar.reshape(1, -1) @ k_inv @ y_bar.reshape(-1, 1)

#         # matrix determinant lemma
#         # https://en.wikipedia.org/wiki/Matrix_determinant_lemma
#         chol_m = jnp.linalg.cholesky(k_mm)
#         right = jsp.linalg.solve_triangular(chol_m, k_nm.T, lower=True)  # m x n
#         term = (right / noise_n.reshape(1, -1)) @ right.T + jnp.eye(X_inducing.shape[0])
#         log_det_term = jnp.log(jnp.linalg.cholesky(term).diagonal()).sum() * 2
#         log_det_noise = jnp.log(noise_n).sum()
#         complexity_penalty = log_det_term + log_det_noise

#         # trace
#         k_diag = (jax.vmap(lambda x: kernel_fn(x.reshape(1, -1), x.reshape(1, -1)))(X)).reshape(-1)
#         q_diag = jnp.square(right).sum(axis=0)
#         trace_term = ((k_diag - q_diag) / noise_n).sum()

#         # print(
#         #     "data fit",
#         #     data_fit,
#         #     "complexity penalty",
#         #     complexity_penalty + X.shape[0] * jnp.log(2 * jnp.pi),
#         #     "trace term",
#         #     trace_term,
#         # )
#         log_prob = -(0.5 * (data_fit + complexity_penalty + trace_term + X.shape[0] * jnp.log(2 * jnp.pi))).squeeze()
#         if include_prior:
#             return log_prob + prior_log_prob
#         else:
#             return log_prob

#     def predict(self, params, X, y, X_test, return_cov=True, include_noise=True):
#         X_inducing = params["X_inducing"]
#         kernel_fn = self.kernel(params)

#         if self.mean.__class__.__name__ == "ZeroMean":
#             mean = y.mean()
#         else:
#             mean = self.mean(params)

#         if self.kernel.__class__.__name__ == "HeinonenGibbsKernel":
#             k_mm = self.kernel.train_cov(params, return_prior_log_prob=False)
#         else:
#             k_mm = kernel_fn(X_inducing, X_inducing)

#         y_bar = y - mean
#         k_mm = k_mm + jnp.eye(X_inducing.shape[0]) * jnp.jitter
#         chol_m = jnp.linalg.cholesky(k_mm)
#         k_nm = kernel_fn(X, X_inducing)
#         noise_n = self.noise(params, X).squeeze()

#         chol_m_inv_mn = jsp.linalg.solve_triangular(chol_m, k_nm.T, lower=True)  # m x n

#         chol_m_inv_mn_by_noise = chol_m_inv_mn / noise_n.reshape(1, -1)
#         A = chol_m_inv_mn_by_noise @ chol_m_inv_mn.T + jnp.eye(X_inducing.shape[0])
#         prod_y_bar = chol_m_inv_mn_by_noise @ y_bar
#         chol_A = jnp.linalg.cholesky(A)
#         post_mean = chol_m @ jsp.linalg.cho_solve((chol_A, True), prod_y_bar)

#         k_new_m = kernel_fn(X_test, X_inducing)
#         K_inv_y = jsp.linalg.cho_solve((chol_m, True), post_mean)
#         pred_mean = k_new_m @ K_inv_y + mean
#         if return_cov:
#             k_new_new = kernel_fn(X_test, X_test)
#             k_inv_new = jsp.linalg.cho_solve((chol_m, True), k_new_m.T)
#             posterior_cov = k_new_new - k_new_m @ k_inv_new

#             chol_A_ = jnp.linalg.cholesky(chol_m @ A @ chol_m.T)
#             subspace_cov = k_new_m @ jsp.linalg.cho_solve((chol_A_, True), k_new_m.T)

#             pred_cov = posterior_cov + subspace_cov
#             if include_noise:
#                 pred_cov = self.add_noise(pred_cov, self.noise(params, X_test))
#             return pred_mean, pred_cov
#         else:
#             return pred_mean

#     def __initialize_params__(self, key, X, X_inducing):
#         if X_inducing is None:
#             assert self.X_inducing is not None, "X_inducing must be specified."
#             X_inducing = self.X_inducing
#         return {"X_inducing": X_inducing}

#     def __get_bijectors__(self):
#         return {"X_inducing": Identity()}

#     def __get_priors__(self):
#         return {"X_inducing": None}
