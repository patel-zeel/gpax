from __future__ import annotations
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from jax_tqdm import scan_tqdm

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from gpax.core import Module, Parameter, get_default_jitter, get_positive_bijector
from gpax.utils import add_to_diagonal, get_a_inv_b, repeat_to_size

import matplotlib.pyplot as plt

import jaxopt
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
        sparse=False,
    ):
        super(LatentGP, self).__init__()
        self.vmap = vmap
        self.X_inducing = X_inducing
        self.sparse = sparse

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

        value = jnp.asarray(value)
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


class LatentGPGaussianBasis(LatentGP):
    """
    This method does not have inducing points. Pass full data X to X_inducing argument.
    """

    def __init__(self, X_inducing, grid_size=None, vmap=False, sparse=None, active_dims=None):
        super(LatentGPGaussianBasis, self).__init__(X_inducing, kernel=None, vmap=vmap)
        assert active_dims is not None, "active_dims must be specified."
        self.active_dims = active_dims
        grid_size = 10 if grid_size is None else grid_size
        upper = X_inducing.max(axis=0)
        lower = X_inducing.min(axis=0)
        self.vmap = vmap
        self.X_grid = jax.vmap(lambda l, u: jnp.linspace(l, u, grid_size), out_axes=1)(lower, upper)
        self.scale = (upper - lower) / (grid_size - 1)

        self.theta = Parameter(
            jnp.linspace(0.2, 0.8, grid_size).reshape(-1, 1).repeat(X_inducing.shape[1], axis=1),
            bijector=get_positive_bijector(),
            prior=tfd.Uniform(low=0.0, high=1 / grid_size),
        )

        # n_repeat = len(self.active_dims) if self.vmap else 1
        # self.bias = Parameter(jnp.array(1.0).repeat(n_repeat), bijector=get_positive_bijector())

    def __call__(self, X_inducing):
        def predict_fn(X):
            def predict_1d(x, x_grid, scale, theta):
                pred_y = jax.vmap(lambda x: jsp.stats.norm.pdf(x, loc=x_grid, scale=scale))(x) @ theta
                return pred_y

            pred_y_fn = jax.vmap(predict_1d, in_axes=(1, 1, 0, 1), out_axes=1)
            # vec_bias = self.bias() if self.vmap else self.bias().repeat(len(self.active_dims))
            pred_y_dim = pred_y_fn(X[:, self.active_dims], self.X_grid, self.scale, self.theta())
            if self.vmap:
                pred_y = pred_y_dim
            else:
                pred_y = pred_y_dim.sum(axis=1)

            if self._training:
                return pred_y, jnp.array(0.0)
            else:
                return pred_y

        return predict_fn


class LatentGPHeinonen(LatentGP):
    def __call__(self, X_inducing):
        if not self.sparse:
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
    def fit(
        self, key, X, y, customize_fn=lambda x: None, lr=0.01, epochs=100, initialize_params=True, optimizer_name="adam"
    ):
        if initialize_params:
            self.initialize(key)

        customize_fn(self)

        init_raw_params = self.get_raw_parameters()

        def loss_fn(raw_params):
            self.set_raw_parameters(raw_params)
            return -self.log_probability(X, y)

        if optimizer_name == "adam":
            optimizer = optax.adam(learning_rate=lr)
            init_state = optimizer.init(init_raw_params)

            value_and_grad_fn = jax.value_and_grad(loss_fn)

            @jax.jit
            # @scan_tqdm(epochs) # did not work
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
        elif optimizer_name in ["lbfgsb", "bfgs"]:
            methods = {"lbfgsb": "L-BFGS-B", "bfgs": "BFGS"}
            optimizer = jaxopt.ScipyMinimize(fun=loss_fn, method=methods[optimizer_name], maxiter=epochs)
            solution = optimizer.run(init_raw_params)
            return {"raw_params": solution.params, "loss_history": [solution.state.fun_val]}

    def plot(self, X, y, X_test, ax=None, alpha=0.3, s=20):
        if X.shape[1] > 1:
            raise NotImplementedError("Only 1D inputs are supported")

        if ax is None:
            ax = plt.gca()

        self.eval()
        X_inducing = self.X_inducing()
        likelihood_fn = self.likelihood.get_likelihood_fn(X_inducing)

        pred_mean, pred_var = self.predict(X, y, X_test, include_noise=False)
        pred_noise_scale = likelihood_fn(X_test)
        assert jnp.all(pred_noise_scale >= 0.0)

        pred_mean = pred_mean.squeeze()
        pred_std = jnp.sqrt(pred_var)
        pred_noisy_std = jnp.sqrt(pred_var + pred_noise_scale**2)
        ax.scatter(X, y, s=s, label="Observations")
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
            pred_mean - 2 * pred_noisy_std,
            pred_mean + 2 * pred_noisy_std,
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
            self.X_inducing = Parameter(X_inducing)
        else:
            self.X_inducing = lambda: None

        # post process
        assert likelihood.__class__.__name__ in ["Gaussian", "Heteroscedastic"]
        if likelihood.__class__.__name__ == "Heteroscedastic":
            assert likelihood.latent_model.__class__.__name__ in [
                "LatentGPHeinonen",
                "LatentGPPlagemann",
                "LatentGPDeltaInducing",
                "LatentGPGaussianBasis",
            ]
            if likelihood.latent_model.__class__.__name__ == "LatentGPHeinonen":
                self.X_inducing.trainable(False)

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
        noisy_covariance = add_to_diagonal(covariance, noise_scale.ravel() ** 2, 0.0)

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
            print(f"{log_likelihood=}, {log_prior_likelihood=}, {log_prior_kernel=}")
            log_probability = log_likelihood + log_prior_kernel + log_prior_likelihood
            # if hasattr(self.likelihood, "latent_model"):
            #     if self.likelihood.latent_model.__class__.__name__ == "LatentGPGaussianBasis":
            #         log_probability += self.likelihood.latent_model.theta() ** 2
            # TODO: cover this within the latent model
            return log_probability / X.size
        else:
            return log_likelihood / X.size

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

        def predict_fn(X_test, return_cov=True, full_cov=False, include_noise=True):
            K_star = kernel_fn(X_test, X)
            pred_mean = K_star @ k_inv_y + mean

            if return_cov:
                if full_cov:
                    k_inv_k_star = jsp.linalg.cho_solve((k_cholesky, True), K_star.T)
                    pred_cov = kernel_fn(X_test, X_test) - (K_star @ k_inv_k_star)
                    if include_noise:
                        noise_var = likelihood_fn(X_test) ** 2
                        pred_cov = add_to_diagonal(pred_cov, noise_var, 0.0)

                    return pred_mean, pred_cov
                else:

                    def scalar_var_fn(x_test, k_star):
                        k_star = k_star.reshape(1, -1)
                        x_test = x_test.reshape(1, -1)

                        k_inv_k_star = jsp.linalg.cho_solve((k_cholesky, True), k_star.T)
                        pred_var = kernel_fn(x_test, x_test) - (k_star @ k_inv_k_star)
                        return pred_var.squeeze()

                    pred_var = jax.vmap(scalar_var_fn)(X_test, K_star)
                    if include_noise:
                        noise_var = likelihood_fn(X_test) ** 2
                        pred_var = pred_var + noise_var
                    return pred_mean, pred_var
            else:
                return pred_mean

        return predict_fn

    def predict(self, X, y, X_test, return_cov=True, full_cov=False, include_noise=True):
        """
        This method is suitable for one time prediction.
        In case of batch prediction, it is better to use `condition` method in combination with `predict`.
        """
        predict_fn = self.condition(X, y)
        return predict_fn(X_test, return_cov=return_cov, full_cov=full_cov, include_noise=include_noise)

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


class SparseGPRegression(GP):
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

        return (log_prob + log_prior_kernel + log_prior_likelihood) / X.size

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

        def predict_fn(X_test, return_cov=True, full_cov=False, include_noise=True):
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

            chol_A_ = jnp.linalg.cholesky(chol_m @ A @ chol_m.T)

            if return_cov:
                if full_cov:
                    k_new_new = kernel_fn(X_test, X_test)
                    k_inv_new = jsp.linalg.cho_solve((chol_m, True), k_new_m.T)
                    posterior_cov = k_new_new - k_new_m @ k_inv_new

                    subspace_cov = k_new_m @ jsp.linalg.cho_solve((chol_A_, True), k_new_m.T)

                    pred_cov = posterior_cov + subspace_cov
                    if include_noise:
                        pred_noise_scale = likelihood_fn(X_test)
                        pred_cov = add_to_diagonal(pred_cov, pred_noise_scale**2, get_default_jitter())
                    return pred_mean, pred_cov
                else:

                    def scalar_var_fn(k_new_m_single, x_test):
                        x_test = x_test.reshape(1, -1)
                        k_new_m_single = k_new_m_single.reshape(1, -1)

                        k_new_new = kernel_fn(x_test, x_test)
                        k_inv_new = jsp.linalg.cho_solve((chol_m, True), k_new_m_single.T)
                        posterior_var = k_new_new - k_new_m_single @ k_inv_new

                        subspace_var = k_new_m_single @ jsp.linalg.cho_solve((chol_A_, True), k_new_m_single.T)

                        pred_var = posterior_var + subspace_var
                        return pred_var.squeeze()

                    pred_var = jax.vmap(scalar_var_fn)(k_new_m, X_test)
                    if include_noise:
                        pred_noise_scale = likelihood_fn(X_test)
                        pred_var = pred_var + pred_noise_scale**2
                    return pred_mean, pred_var

            else:
                return pred_mean

        return predict_fn

    def predict(self, X, y, X_test, return_cov=True, full_cov=False, include_noise=True):
        predict_fn = self.condition(X, y)
        return predict_fn(X_test, return_cov=return_cov, full_cov=full_cov, include_noise=include_noise)
