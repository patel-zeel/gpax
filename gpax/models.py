from __future__ import annotations
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from gpax.core import Module, Parameter, get_default_jitter, get_positive_bijector
from gpax.utils import add_to_diagonal, get_a_inv_b, train_fn

from jaxtyping import Array, Float
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpax.kernels import Kernel
    from gpax.likelihoods import Likelihood
    from gpax.means import Mean


class Model(Module):
    pass


class LatentGP(Model):
    def __init__(
        self,
        X: Float[Array, "N D"],
        lengthscale: float = 1.0,
        scale: float = 1.0,
        latent_kernel_type: Kernel = None,
        vmap: bool = False,
    ):
        super(LatentGP, self).__init__()
        self.latent_kernel_type = latent_kernel_type
        self.vmap = vmap

        self.lengthscale = Parameter(jnp.array(lengthscale).repeat(X.shape[1]))
        if self.vmap is False:
            self.latent = Parameter(jnp.zeros(X.shape[0]))
            self.scale = Parameter(scale)
        else:
            self.latent = Parameter(jnp.zeros(X.shape))
            self.scale = Parameter(jnp.ones(X.shape[1]))

    def common(self, ls, scale, latent, X):
        kernel = self.latent_kernel_type(X=X, ARD=True, lengthscale=ls, scale=scale).eval().get_kernel_fn()
        cov = kernel(X, X)
        noisy_cov = add_to_diagonal(cov, 0.0, get_default_jitter())
        chol = jnp.linalg.cholesky(noisy_cov)
        log_fx = chol @ latent
        return log_fx, chol, kernel

    def __call__(self, X_inducing, X, X_new=None):
        if self.vmap is False:
            return self.vmap_fn(self.lengthscale(), self.scale(), self.latent(), X_inducing, X, X_new)
        else:
            if self.training:
                return jax.vmap(self.vmap_fn, in_axes=(0, 0, 1, 1, 1), out_axes=(1, 0))(
                    self.lengthscale(),
                    self.scale(),
                    self.latent(),
                    X_inducing[..., None],
                    X[..., None],
                )
            else:
                return jax.vmap(self.vmap_fn, in_axes=(0, 0, 1, 1, 1, 1), out_axes=(1, 1))(
                    self.lengthscale(),
                    self.scale(),
                    self.latent(),
                    X_inducing[..., None],
                    X[..., None],
                    X_new[..., None],
                )


class LatentGPHeinonen(LatentGP):
    def vmap_fn(self, ls, scale, latent, X_inducing, X, X_new=None):
        # X_inducing and X must be same in this method
        positive_bijector = get_positive_bijector()

        log_fx, chol, kernel = self.common(ls, scale, latent, X_inducing)
        fx = positive_bijector(log_fx)
        if self.training:
            log_prior = tfd.MultivariateNormalTriL(loc=log_fx.mean(), scale_tril=chol).log_prob(log_fx)
            return fx, log_prior
        else:

            def predict_fn(x):
                cross_cov = kernel(x, X_inducing)
                alpha = jsp.linalg.cho_solve((chol, True), log_fx)
                fx_new = positive_bijector(cross_cov @ alpha)
                return fx_new

            fx = predict_fn(X)
            fx_new = predict_fn(X_new)
            return fx, fx_new


class LatentGPDeltaInducing(LatentGP):
    def vmap_fn(self, ls, scale, latent, X_inducing, X, X_new=None):
        positive_bijector = get_positive_bijector()
        log_fx_inducing, chol_inducing, kernel = self.common(ls, scale, latent, X_inducing)
        cross_cov = kernel(X, X_inducing)
        alpha = jsp.linalg.cho_solve((chol_inducing, True), log_fx_inducing)
        log_fx = cross_cov @ alpha
        fx = positive_bijector(log_fx)
        if self.training:
            fx = positive_bijector(log_fx)
            log_prior = tfd.MultivariateNormalTriL(loc=log_fx_inducing.mean(), scale_tril=chol_inducing).log_prob(
                log_fx_inducing
            )
            return fx, log_prior
        else:
            cross_cov_new = kernel(X_new, X_inducing)
            fx_new = positive_bijector(cross_cov_new @ alpha)
            return fx, fx_new


class ExactGPRegression(Model):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        mean: Mean,
        X_inducing: Float[Array, "N D"] = None,
    ):
        super(ExactGPRegression, self).__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean = mean
        if X_inducing is not None:
            self.X_inducing = Parameter(X_inducing)
        else:
            self.X_inducing = None

    def log_probability(self, X, y):
        self.train()
        """
        prior_type: default: None, possible values: "prior", "posterior", None
        """
        kernel_fn = self.kernel.get_kernel_fn(self.X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(self.X_inducing)

        covariance, log_prior_cov = kernel_fn(X, X)
        noise_scale, log_prior_noise = likelihood_fn(X)
        noisy_covariance = add_to_diagonal(covariance, noise_scale**2, get_default_jitter())

        log_likelihood = tfd.MultivariateNormalFullCovariance(
            loc=self.mean(y=y), covariance_matrix=noisy_covariance
        ).log_prob(y)

        return log_likelihood + log_prior_cov + log_prior_noise

    def condition(self, X, y):
        """
        This function is useful while doing batch prediction.
        """
        self.eval()

        kernel_fn = self.kernel.get_kernel_fn(self.X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(self.X_inducing)

        train_cov = kernel_fn(X, X)
        noise_scale = likelihood_fn(X)

        mean = self.mean(y=y)
        y_bar = y - mean
        noisy_covariance = add_to_diagonal(train_cov, noise_scale**2, get_default_jitter())
        k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)

        def predict_fn(X_test, return_cov=True, include_noise=True):
            K_star = kernel_fn(X_test, X)
            pred_mean = K_star @ k_inv_y + mean

            if return_cov:
                k_inv_k_star = jsp.linalg.cho_solve((k_cholesky, True), K_star.T)
                pred_cov = kernel_fn(X_test, X_test) - (K_star @ k_inv_k_star)
                if include_noise:
                    pred_cov = add_to_diagonal(pred_cov, likelihood_fn(X_test) ** 2, get_default_jitter())
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
#         if self.noise.__class__.__name__ == "HeteroscedasticHeinonenNoise":
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
