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

import matplotlib.pyplot as plt

import optax

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

    def reverse_init(self, value):
        pos_bijector = get_positive_bijector()
        kernel_fn = self.kernel.eval().get_kernel_fn()

        def out_vmap_fn(X_inducing, raw_value):
            cov = kernel_fn(X_inducing, X_inducing)
            noisy_cov = add_to_diagonal(cov, 0.0, get_default_jitter())
            chol = jnp.linalg.cholesky(noisy_cov)
            latent = jsp.linalg.solve_triangular(chol, raw_value, lower=True)
            # latent = jnp.linalg.solve(chol, raw_value)
            return latent

        # value = jnp.asarray(value)
        assert value.size in (1, self.latent().size)
        if value.size == 1:
            value = jnp.ones(self.latent().shape) * value
        else:
            value = value.reshape(self.latent().shape)
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
            log_fx_inducing = chol @ latent
            return log_fx_inducing, chol

        if self.vmap:
            out_vmap_fn = jax.vmap(out_vmap_fn, in_axes=(1, 1), out_axes=(1, 2))
            log_fx_inducing, chol = out_vmap_fn(X_inducing[..., None], self.latent())
        else:
            log_fx_inducing, chol = out_vmap_fn(X_inducing, self.latent())

        def predict_fn(X):
            if self._training:

                def in_vmap_fn(x, log_fx_inducing, chol):
                    log_prior = tfd.MultivariateNormalTriL(loc=log_fx_inducing.mean(), scale_tril=chol).log_prob(
                        log_fx_inducing
                    )
                    fx = pos_bijector(log_fx_inducing)
                    return fx, log_prior

                if self.vmap:
                    X = X[..., None]
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=(1, 0))
            else:

                def in_vmap_fn(x, log_fx_inducing, chol):
                    mean = log_fx_inducing.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx_inducing - mean)
                    return pos_bijector(mean + cross_cov @ alpha)

                if self.vmap:
                    X = X[..., None]
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=1)

            return in_vmap_fn(X, log_fx_inducing, chol)

        return predict_fn


class LatentGPDeltaInducing(LatentGP):
    def __call__(self, X_inducing):
        pos_bijector = get_positive_bijector()
        kernel_fn = self.kernel.eval().get_kernel_fn()

        def out_vmap_fn(X_inducing, latent):
            cov = kernel_fn(X_inducing, X_inducing)
            noisy_cov = add_to_diagonal(cov, 0.0, get_default_jitter())
            chol = jnp.linalg.cholesky(noisy_cov)
            log_fx_inducing = chol @ latent
            return log_fx_inducing, chol

        if self.vmap:
            out_vmap_fn = jax.vmap(out_vmap_fn, in_axes=(1, 1), out_axes=(1, 2))
            log_fx_inducing, chol = out_vmap_fn(X_inducing[..., None], self.latent())
        else:
            log_fx_inducing, chol = out_vmap_fn(X_inducing, self.latent())

        def predict_fn(X):
            if self._training:

                def in_vmap_fn(x, log_fx_inducing, chol):
                    log_prior = tfd.MultivariateNormalTriL(loc=log_fx_inducing.mean(), scale_tril=chol).log_prob(
                        log_fx_inducing
                    )
                    mean = log_fx_inducing.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx_inducing - mean)
                    fx = pos_bijector(mean + cross_cov @ alpha)
                    return fx, log_prior

                if self.vmap:
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=(1, 0))
                    return in_vmap_fn(X[..., None], log_fx_inducing, chol)
                return in_vmap_fn(X, log_fx_inducing, chol)

            else:

                def in_vmap_fn(x, log_fx_inducing, chol):
                    mean = log_fx_inducing.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx_inducing - mean)
                    return pos_bijector(mean + cross_cov @ alpha)

                if self.vmap:
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=1)
                    return in_vmap_fn(X[..., None], log_fx_inducing, chol)
                return in_vmap_fn(X, log_fx_inducing, chol)

        return predict_fn


class LatentGPPlagemann(LatentGP):
    def __call__(self, X_inducing):
        pos_bijector = get_positive_bijector()
        kernel_fn = self.kernel.eval().get_kernel_fn()

        def out_vmap_fn(X_inducing, latent):
            cov = kernel_fn(X_inducing, X_inducing)
            noisy_cov = add_to_diagonal(cov, 0.0, get_default_jitter())
            chol = jnp.linalg.cholesky(noisy_cov)
            log_fx_inducing = chol @ latent
            return log_fx_inducing, chol

        if self.vmap:
            out_vmap_fn = jax.vmap(out_vmap_fn, in_axes=(1, 1), out_axes=(1, 2))
            log_fx_inducing, chol = out_vmap_fn(X_inducing[..., None], self.latent())
        else:
            log_fx_inducing, chol = out_vmap_fn(X_inducing, self.latent())

        def predict_fn(X):
            if self._training:

                def in_vmap_fn(x, log_fx_inducing_inducing, chol):
                    mean = log_fx_inducing_inducing.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx_inducing_inducing - mean)
                    v = jsp.linalg.cho_solve((chol, True), cross_cov.T)
                    test_cov = kernel_fn(x, x)
                    pred_cov = test_cov - cross_cov @ v
                    pred_cov = add_to_diagonal(pred_cov, 0.0, get_default_jitter())

                    log_fx = mean + cross_cov @ alpha
                    fx = pos_bijector(log_fx)

                    log_prior = tfd.MultivariateNormalFullCovariance(log_fx, pred_cov).log_prob(log_fx)
                    # log_prior = -jnp.log(jnp.linalg.cholesky(pred_cov).diagonal()).sum()

                    return fx, log_prior

                if self.vmap:
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=(1, 0))
                    return in_vmap_fn(X[..., None], log_fx_inducing, chol)
                return in_vmap_fn(X, log_fx_inducing, chol)

            else:

                def in_vmap_fn(x, log_fx_inducing, chol):
                    mean = log_fx_inducing.mean()
                    cross_cov = kernel_fn(x, X_inducing)
                    alpha = jsp.linalg.cho_solve((chol, True), log_fx_inducing - mean)
                    return pos_bijector(mean + cross_cov @ alpha)

                if self.vmap:
                    in_vmap_fn = jax.vmap(in_vmap_fn, in_axes=(1, 1, 2), out_axes=1)
                    return in_vmap_fn(X[..., None], log_fx_inducing, chol)
                return in_vmap_fn(X, log_fx_inducing, chol)

        return predict_fn


class GP(Model):
    def plot(self, X, y, X_test, ax=None, alpha=0.3):
        if X.shape[1] > 1:
            raise NotImplementedError("Only 1D inputs are supported")

        if ax is None:
            ax = plt.gca()

        self.eval()
        X_inducing = self.X_inducing()
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        pred_mean, pred_cov = self.predict(X, y, X_test, include_noise=False)
        pred_noise_scale = likelihood_fn(X_test)
        assert pred_noise_scale > 0.0

        pred_mean = pred_mean.squeeze()
        pred_std = jnp.sqrt(jnp.diag(pred_cov))
        ax.scatter(X, y, label="Observations")
        ax.plot(X_test, pred_mean, label="Posterior mean")
        ax.fill_between(
            X_test.ravel(),
            pred_mean - 2 * pred_std,
            pred_mean + 2 * pred_std,
            alpha=alpha,
            label="95% CI",
        )
        ax.fill_between(
            X_test.ravel(),
            pred_mean - 2 * pred_std - 2 * pred_noise_scale,
            pred_mean + 2 * pred_std + 2 * pred_noise_scale,
            alpha=alpha,
            label="95% CI + noise",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper right")

        return ax


# @jtu.register_pytree_node_class
class ExactGPRegression(GP):
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

    def log_probability(self, X, y, include_prior=True):
        self.train()  # Set model to train mode
        """
        prior_type: default: None, possible values: "prior", "posterior", None
        """
        X_inducing = self.X_inducing()

        kernel_fn = self.kernel.get_kernel_fn(X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        covariance, log_prior_kernel = kernel_fn(X, X)

        noise_scale, log_prior_likelihood = likelihood_fn(X)
        noisy_covariance = add_to_diagonal(covariance, noise_scale**2, 0.0)

        log_likelihood = tfd.MultivariateNormalFullCovariance(
            loc=self.mean(y=y), covariance_matrix=noisy_covariance
        ).log_prob(y)

        # y_bar = y - self.mean(y=y)
        # k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)
        # fit_term = -0.5 * (y_bar.ravel() * k_inv_y.ravel()).sum()  # This will break for multi-dimensional y
        # penalty_term = -jnp.log(k_cholesky.diagonal()).sum()
        # log_likelihood = fit_term + penalty_term - 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)

        # lpd = tfd.Normal(self.mean(y=y), jnp.sqrt(jnp.diagonal(noisy_covariance))).log_prob(y).mean()
        # print(f"{fit_term=}, {penalty_term=}, {log_likelihood=}, {lpd=}")
        # print(
        #     f"{log_likelihood=:.20f}, {log_prior_likelihood=}, total={log_likelihood + log_prior_kernel + log_prior_likelihood}"
        # )  # DEBUG
        if include_prior:
            return log_likelihood + log_prior_kernel + log_prior_likelihood
        else:
            return log_likelihood

    def fit(self, key, X, y, lr=0.01, epochs=100):
        self.initialize(key)
        init_raw_params = self.get_raw_parameters()
        optimizer = optax.adam(learning_rate=lr)
        init_state = optimizer.init(init_raw_params)

        def loss_fn(raw_params):
            self.set_raw_parameters(raw_params)
            return -self.log_probability(X, y)

        value_and_grad_fn = jax.value_and_grad(loss_fn)

        def one_step(raw_params_and_state, aux):
            raw_params, state = raw_params_and_state
            loss, grads = value_and_grad_fn(raw_params)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            return (raw_params, state), (raw_params, loss)

        (raw_params, state), (raw_params_history, loss_history) = jax.lax.scan(
            f=one_step, init=(init_raw_params, init_state), xs=None, length=epochs
        )

        self.set_raw_parameters(raw_params)
        return {
            "raw_params": raw_params,
            "raw_params_history": raw_params_history,
            "loss_history": loss_history,
        }

    def nlpd(self, X, y):
        self.train()
        """
        prior_type: default: None, possible values: "prior", "posterior", None
        """
        X_inducing = self.X_inducing()

        kernel_fn = self.kernel.get_kernel_fn(X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        covariance, log_kernel_prior = kernel_fn(X, X)

        noise_scale, log_likelihood_prior = likelihood_fn(X)
        noisy_covariance = add_to_diagonal(covariance, noise_scale**2, 0.0)

        # log_likelihood = tfd.MultivariateNormalFullCovariance(
        #     loc=self.mean(y=y), covariance_matrix=noisy_covariance
        # ).log_prob(y)

        # y_bar = y - self.mean(y=y)
        # k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)
        # fit_term = -0.5 * (y_bar.ravel() * k_inv_y.ravel()).sum()  # This will break for multi-dimensional y
        # penalty_term = -jnp.log(k_cholesky.diagonal()).sum()
        # log_likelihood = fit_term + penalty_term - 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)

        nlpd = -tfd.Normal(self.mean(y=y), jnp.sqrt(jnp.diagonal(noisy_covariance))).log_prob(y).mean()
        # print(f"{fit_term=}, {penalty_term=}, {log_likelihood=}, {lpd=}")
        # print(
        #     f"{log_likelihood=:.20f}, {log_prior_likelihood=}, total={log_likelihood + log_prior_kernel + log_prior_likelihood}"
        # )  # DEBUG
        return nlpd

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

        def predict_fn(X_test, return_cov, include_noise):
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

    def tree_flatten(self):
        raw_params = self.get_raw_parameters()
        flat_params, treedef = jtu.tree_flatten(raw_params)
        return flat_params, (treedef, self.kernel, self.likelihood, self.mean)

    @classmethod
    def tree_unflatten(cls, aux_data, flat_params):
        (treedef, kernel, likelihood, mean) = aux_data
        # obj =
        # raw_params = jtu.tree_unflatten(treedef, flat_params)
        # obj.set_raw_parameters(raw_params)
        obj = cls(kernel, likelihood, mean)
        obj.kernel.variance.set_raw_value(flat_params[1])
        return obj


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

        _, log_prior_likelihood = likelihood_fn(X)
        noise_n = self.likelihood.eval().get_likelihood_fn(X_inducing)(X) ** 2
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

    def nlpd(self, X, y):
        self.train()  # Set model to training mode

        X_inducing = self.X_inducing()

        kernel_fn = self.kernel.get_kernel_fn(X_inducing)
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        _, log_prior_likelihood = likelihood_fn(X)
        noise_n = self.likelihood.eval().get_likelihood_fn(X_inducing)(X) ** 2
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

        raise NotImplementedError

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
