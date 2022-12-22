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
from gpax.utils import add_to_diagonal, get_a_inv_b, repeat_to_size

from jaxtyping import Array, Float
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpax.kernels import Kernel
    from gpax.likelihoods import Likelihood
    from gpax.means import Mean

is_parameter = lambda x: isinstance(x, Parameter)


class Model(Module):
    pass


class LatentModel(Model):
    pass


class LatentGP(LatentModel):
    def __init__(
        self,
        X_inducing: Array,
        kernel: Kernel,
        vmap: bool = False,
    ):
        super(LatentGP, self).__init__()
        self.vmap = vmap
        self.X_inducing = X_inducing

        self.kernel = kernel

        if self.vmap:
            self.latent = Parameter(jnp.ones(X_inducing.shape))
        else:
            self.latent = Parameter(jnp.ones(X_inducing.shape[0]))

    def reverse_init_latent(self, value):
        pos_bijector = get_positive_bijector()
        kernel_fn = self.kernel.eval().get_kernel_fn()

        def out_vmap_fn(X_inducing, raw_value):
            cov = kernel_fn(X_inducing, X_inducing)
            noisy_cov = add_to_diagonal(cov, 0.0, get_default_jitter())
            chol = jnp.linalg.cholesky(noisy_cov)
            latent = jsp.linalg.solve_triangular(chol, raw_value, lower=True)
            # latent = jnp.linalg.solve(chol, raw_value)
            return latent

        assert value.size == 1
        value = jnp.ones(self.latent().shape) * value
        raw_value = pos_bijector.inverse(value)

        if self.vmap:
            out_vmap_fn = jax.vmap(out_vmap_fn, in_axes=(1, 1), out_axes=1)
            latent = out_vmap_fn(self.X_inducing[..., None], raw_value)
            self.latent.set_raw_value(latent)
        else:
            latent = out_vmap_fn(self.X_inducing, raw_value)
            self.latent.set_raw_value(latent)


class LatentGPHeinonen(LatentGP):
    def __call__(self, X_inducing):
        X_inducing = jax.lax.stop_gradient(X_inducing)
        pos_bijector = get_positive_bijector()
        kernel_fn = self.kernel.eval().get_kernel_fn()

        def out_vmap_fn(X_inducing, latent):
            cov = kernel_fn(X_inducing, X_inducing)
            noisy_cov = add_to_diagonal(cov, 0.0, get_default_jitter())
            chol = jnp.linalg.cholesky(noisy_cov)
            log_fx = chol @ latent
            return log_fx, chol

        if self.vmap:
            out_vmap_fn = jax.vmap(out_vmap_fn, in_axes=(1, 1), out_axes=(1, 2))
            log_fx, chol = out_vmap_fn(X_inducing[..., None], self.latent())
        else:
            log_fx, chol = out_vmap_fn(X_inducing, self.latent())

        def predict_fn(X):
            if self.training:

                def in_vmap_fn(x, log_fx, chol):
                    log_prior = tfd.MultivariateNormalTriL(loc=log_fx.mean(), scale_tril=chol).log_prob(log_fx)
                    fx = pos_bijector(log_fx)
                    return fx, log_prior

                if self.vmap:
                    X = X[..., None]
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=(1, 0))
            else:

                def in_vmap_fn(x, log_fx, chol):
                    mean = log_fx.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx - mean)
                    return pos_bijector(mean + cross_cov @ alpha)

                if self.vmap:
                    X = X[..., None]
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=1)

            return in_vmap_fn(X, log_fx, chol)

        return predict_fn


class LatentGPDeltaInducing(LatentGP):
    def __call__(self, X_inducing):
        pos_bijector = get_positive_bijector()
        kernel_fn = self.kernel.eval().get_kernel_fn()

        def out_vmap_fn(X_inducing, latent):
            cov = kernel_fn(X_inducing, X_inducing)
            noisy_cov = add_to_diagonal(cov, 0.0, get_default_jitter())
            chol = jnp.linalg.cholesky(noisy_cov)
            log_fx = chol @ latent
            return log_fx, chol

        if self.vmap:
            out_vmap_fn = jax.vmap(out_vmap_fn, in_axes=(1, 1), out_axes=(1, 2))
            log_fx, chol = out_vmap_fn(X_inducing[..., None], self.latent())
        else:
            log_fx, chol = out_vmap_fn(X_inducing, self.latent())

        def predict_fn(X):
            if self.training:

                def in_vmap_fn(x, log_fx, chol):
                    log_prior = tfd.MultivariateNormalTriL(loc=log_fx.mean(), scale_tril=chol).log_prob(log_fx)
                    mean = log_fx.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx - mean)
                    fx = pos_bijector(mean + cross_cov @ alpha)
                    return fx, log_prior

                if self.vmap:
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=(1, 0))
                    return in_vmap_fn(X[..., None], log_fx, chol)
                return in_vmap_fn(X, log_fx, chol)

            else:

                def in_vmap_fn(x, log_fx, chol):
                    mean = log_fx.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx - mean)
                    return pos_bijector(mean + cross_cov @ alpha)

                if self.vmap:
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=1)
                    return in_vmap_fn(X[..., None], log_fx, chol)
                return in_vmap_fn(X, log_fx, chol)

        return predict_fn


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
            self.X_inducing = Parameter(X_inducing, fixed_init=True)
        else:
            self.X_inducing = lambda: None

    def log_probability(self, X, y):
        self.train()
        """
        prior_type: default: None, possible values: "prior", "posterior", None
        """
        X_inducing = self.X_inducing()

        kernel_fn = self.kernel.get_kernel_fn(X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        covariance, log_prior_kernel = kernel_fn(X, X)

        noise_scale, log_prior_likelihood = likelihood_fn(X)
        noisy_covariance = add_to_diagonal(covariance, noise_scale**2, 0.0)

        # log_likelihood = tfd.MultivariateNormalFullCovariance(
        #     loc=self.mean(y=y), covariance_matrix=noisy_covariance
        # ).log_prob(y)

        y_bar = y - self.mean(y=y)
        k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)
        log_likelihood = (
            -0.5 * (y_bar.ravel() * k_inv_y.ravel()).sum()  # This will break for multi-dimensional y
            - jnp.log(k_cholesky.diagonal()).sum()
            - 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)
        )
        # print(
        #     f"{log_likelihood=:.20f}, {log_prior_likelihood=}, total={log_likelihood + log_prior_kernel + log_prior_likelihood}"
        # )  # DEBUG
        return log_likelihood + log_prior_kernel + log_prior_likelihood

    def condition(self, X, y):
        """
        This function is useful while doing batch prediction.
        """
        self.eval()

        X_inducing = self.X_inducing()

        kernel_fn = self.kernel.get_kernel_fn(X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        train_cov = kernel_fn(X, X)
        noise_scale = likelihood_fn(X)

        mean = self.mean(y=y)
        y_bar = y - mean
        noisy_covariance = add_to_diagonal(train_cov, noise_scale**2, 0.0)
        k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)

        def predict_fn(X_test, return_cov=True, include_noise=True):
            K_star = kernel_fn(X_test, X)
            pred_mean = K_star @ k_inv_y + mean

            if return_cov:
                k_inv_k_star = jsp.linalg.cho_solve((k_cholesky, True), K_star.T)
                pred_cov = kernel_fn(X_test, X_test) - (K_star @ k_inv_k_star)
                if include_noise:
                    pred_cov = add_to_diagonal(pred_cov, likelihood_fn(X_test) ** 2, 0.0)
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


class SparseGPRegression(Model):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        mean: Mean,
        X_inducing: Float[Array, "N D"],
    ):
        super(SparseGPRegression, self).__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean = mean
        self.X_inducing = Parameter(X_inducing, fixed_init=True)

    def log_probability(self, X, y):
        self.train()  # Set model to training mode

        X_inducing = self.X_inducing()

        kernel_fn = self.kernel.get_kernel_fn(X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        noise_scale, log_prior_likelihood = likelihood_fn(X)
        noise_n = repeat_to_size(noise_scale**2, y.shape[0])
        mean = self.mean(y=y)

        k_mm, log_prior_kernel = kernel_fn(X_inducing, X_inducing)

        y_bar = y - mean
        k_mm = add_to_diagonal(k_mm, 0.0, get_default_jitter())
        pred_kernel_fn = self.kernel.eval().get_kernel_fn(X_inducing)
        k_nm = pred_kernel_fn(X, X_inducing)

        # woodbury identity
        left = k_nm / noise_n.reshape(-1, 1)
        right = left.T
        middle = k_mm + right @ k_nm
        k_inv = jnp.diag(1 / noise_n.squeeze()) - left @ jsp.linalg.cho_solve(
            (jnp.linalg.cholesky(middle), True), right
        )
        data_fit = y_bar.reshape(1, -1) @ k_inv @ y_bar.reshape(-1, 1)

        # matrix determinant lemma
        # https://en.wikipedia.org/wiki/Matrix_determinant_lemma
        chol_m = jnp.linalg.cholesky(k_mm)
        right = jsp.linalg.solve_triangular(chol_m, k_nm.T, lower=True)  # m x n
        term = (right / noise_n.reshape(1, -1)) @ right.T + jnp.eye(X_inducing.shape[0])
        log_det_term = jnp.log(jnp.linalg.cholesky(term).diagonal()).sum() * 2
        log_det_noise = jnp.log(noise_n).sum()
        complexity_penalty = log_det_term + log_det_noise

        # trace
        k_diag = (jax.vmap(lambda x: kernel_fn(x.reshape(1, -1), x.reshape(1, -1)))(X)).reshape(-1)
        q_diag = jnp.square(right).sum(axis=0)
        trace_term = ((k_diag - q_diag) / noise_n).sum()

        log_prob = -(0.5 * (data_fit + complexity_penalty + trace_term + X.shape[0] * jnp.log(2 * jnp.pi))).squeeze()

        return log_prob + log_prior_kernel + log_prior_likelihood

    def condition(self, X, y):
        self.eval()  # Set model to evaluation mode
        X_inducing = self.X_inducing()

        kernel_fn = self.kernel.get_kernel_fn(X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        noise_scale = likelihood_fn(X)
        noise_n = repeat_to_size(noise_scale**2, y.shape[0])
        mean = self.mean(y=y)

        k_mm = kernel_fn(X_inducing, X_inducing)
        k_mm = add_to_diagonal(k_mm, 0.0, get_default_jitter())

        y_bar = y - mean

        chol_m = jnp.linalg.cholesky(k_mm)

        def predict_fn(X_test, return_cov=True, include_noise=True):
            k_nm = kernel_fn(X, X_inducing)
            chol_m_inv_mn = jsp.linalg.solve_triangular(chol_m, k_nm.T, lower=True)  # m x n
            chol_m_inv_mn_by_noise = chol_m_inv_mn / noise_n.reshape(1, -1)
            A = chol_m_inv_mn_by_noise @ chol_m_inv_mn.T + jnp.eye(X_inducing.shape[0])
            prod_y_bar = chol_m_inv_mn_by_noise @ y_bar
            chol_A = jnp.linalg.cholesky(A)
            post_mean = chol_m @ jsp.linalg.cho_solve((chol_A, True), prod_y_bar)

            k_new_m = kernel_fn(X_test, X_inducing)
            K_inv_y = jsp.linalg.cho_solve((chol_m, True), post_mean)
            pred_mean = k_new_m @ K_inv_y + mean
            if return_cov:
                k_new_new = kernel_fn(X_test, X_test)
                k_inv_new = jsp.linalg.cho_solve((chol_m, True), k_new_m.T)
                posterior_cov = k_new_new - k_new_m @ k_inv_new

                chol_A_ = jnp.linalg.cholesky(chol_m @ A @ chol_m.T)
                subspace_cov = k_new_m @ jsp.linalg.cho_solve((chol_A_, True), k_new_m.T)

                pred_cov = posterior_cov + subspace_cov
                if include_noise:
                    pred_noise_scale = likelihood_fn(X_test)
                    pred_cov = add_to_diagonal(pred_cov, pred_noise_scale**2, get_default_jitter())
                return pred_mean, pred_cov
            else:
                return pred_mean

        return predict_fn

    def predict(self, X, y, X_test, return_cov=True, include_noise=True):
        predict_fn = self.condition(X, y)
        return predict_fn(X_test, return_cov=return_cov, include_noise=include_noise)
