from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from gpax.kernels import RBFKernel
from gpax.means import ScalarMean
from gpax.noises import HomoscedasticNoise
from gpax.distributions import NoPrior
from gpax.bijectors import Identity
from gpax.utils import constrain, unconstrain

from gpax.utils import get_raw_log_prior


class AbstractGP:
    def __init__(self, kernel=RBFKernel(), noise=HomoscedasticNoise(), mean=ScalarMean()):
        self.kernel = kernel
        self.noise = noise
        self.mean = mean

        self.constraints = self.get_bijectors()
        self.priors = self.get_priors()

    @abstractmethod
    def log_probability(self, params, X, y):
        return NotImplementedError("This method must be implemented by a subclass.")

    @abstractmethod
    def predict(self, params, X, X_new, return_cov=True, include_noise=True):
        return NotImplementedError("This method must be implemented by a subclass.")

    def initialise_params(self, key, X, X_inducing=None):
        keys = jax.random.split(key, 4)
        kernels_params = self.kernel.initialise_params(keys[0], X=X, X_inducing=X_inducing)

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

    def constrain(self, params):
        return constrain(params, self.constraints)

    def unconstrain(self, params):
        return unconstrain(params, self.constraints)

    def add_noise(self, K, noise):
        rows, columns = jnp.diag_indices_from(K)
        return K.at[rows, columns].set(K[rows, columns] + noise + jnp.jitter)


class ExactGP(AbstractGP):
    def log_probability(self, params, X, y, include_prior=True):
        prior_log_prob = 0.0
        if self.noise.__class__.__name__ == "HeinonenHeteroscedasticNoise":
            if include_prior:
                noise, tmp_prior_log_prob = self.noise.train_noise(params, return_prior_log_prob=True)
                prior_log_prob += tmp_prior_log_prob
            else:
                noise = self.noise.train_noise(params)
        else:
            noise = self.noise(params, X)
        if self.mean.__class__.__name__ == "ZeroMean":
            mean = y.mean()
        else:
            mean = self.mean(params)
        if self.kernel.__class__.__name__ == "HeinonenGibbsKernel":
            if include_prior:
                covariance, tmp_prior_log_prob = self.kernel.train_cov(params, return_prior_log_prob=True)
                prior_log_prob += tmp_prior_log_prob
            else:
                covariance = self.kernel.train_cov(params)
        else:
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
        return log_likelihood + prior_log_prob

    def log_prior(self, params):
        log_prior = get_raw_log_prior(self.priors, params, self.constraints)
        return ravel_pytree(log_prior)[0].sum()

    def condition(self, params, X, y):
        if self.mean.__class__.__name__ == "ZeroMean":
            mean = y.mean()
        else:
            mean = self.mean(params)
        kernel = self.kernel(params)
        if self.kernel.__class__.__name__ == "HeinonenGibbsKernel":
            covariance = self.kernel.train_cov(params)
        else:
            covariance = kernel(X, X)
        if self.noise.__class__.__name__ == "HeinonenHeteroscedasticNoise":
            noise = self.noise.train_noise(params)
        else:
            noise = self.noise(params, X)

        y_bar = y - mean
        noisy_covariance = self.add_noise(covariance, noise)
        L = jnp.linalg.cholesky(noisy_covariance)
        alpha = jsp.linalg.cho_solve((L, True), y_bar)

        def predict_fn(X_test, return_cov=True, include_noise=True):
            K_star = kernel(X_test, X)
            pred_mean = K_star @ alpha + mean

            if return_cov:
                v = jsp.linalg.cho_solve((L, True), K_star.T)
                pred_cov = kernel(X_test, X_test) - (K_star @ v)
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
    def __init__(self, X_inducing, kernel=RBFKernel(), noise=HomoscedasticNoise(), mean=ScalarMean()):
        super().__init__(kernel, noise, mean)
        self.X_inducing = X_inducing

    def log_probability(self, params, X, y, include_prior=True):
        X_inducing = params["X_inducing"]
        kernel_fn = self.kernel(params)
        prior_log_prob = 0.0
        if self.noise.__class__.__name__ == "HeinonenHeteroscedasticNoise":
            if include_prior:
                _, tmp_prior_log_prob = self.noise.train_noise(params, return_prior_log_prob=True)
                prior_log_prob += tmp_prior_log_prob

        if self.mean.__class__.__name__ == "ZeroMean":
            mean = y.mean()
        else:
            mean = self.mean(params)

        if self.kernel.__class__.__name__ == "HeinonenGibbsKernel":
            if include_prior:
                k_mm, tmp_prior_log_prob = self.kernel.train_cov(params, return_prior_log_prob=True)
                prior_log_prob += tmp_prior_log_prob
            else:
                k_mm = self.kernel.train_cov(params, return_prior_log_prob=False)
        else:
            k_mm = kernel_fn(X_inducing, X_inducing)

        y_bar = y - mean
        noise_n = self.noise(params, X).squeeze()
        k_mm = k_mm + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        k_nm = kernel_fn(X, X_inducing)

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

        # print(
        #     "data fit",
        #     data_fit,
        #     "complexity penalty",
        #     complexity_penalty + X.shape[0] * jnp.log(2 * jnp.pi),
        #     "trace term",
        #     trace_term,
        # )
        log_prob = -(0.5 * (data_fit + complexity_penalty + trace_term + X.shape[0] * jnp.log(2 * jnp.pi))).squeeze()
        if include_prior:
            return log_prob + prior_log_prob
        else:
            return log_prob

    def predict(self, params, X, y, X_test, return_cov=True, include_noise=True):
        X_inducing = params["X_inducing"]
        kernel_fn = self.kernel(params)

        if self.mean.__class__.__name__ == "ZeroMean":
            mean = y.mean()
        else:
            mean = self.mean(params)

        if self.kernel.__class__.__name__ == "HeinonenGibbsKernel":
            k_mm = self.kernel.train_cov(params, return_prior_log_prob=False)
        else:
            k_mm = kernel_fn(X_inducing, X_inducing)

        y_bar = y - mean
        k_mm = k_mm + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        chol_m = jnp.linalg.cholesky(k_mm)
        k_nm = kernel_fn(X, X_inducing)
        noise_n = self.noise(params, X).squeeze()

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
                pred_cov = self.add_noise(pred_cov, self.noise(params, X_test))
            return pred_mean, pred_cov
        else:
            return pred_mean

    def __initialise_params__(self, key, X, X_inducing):
        if X_inducing is None:
            assert self.X_inducing is not None, "X_inducing must be specified."
            X_inducing = self.X_inducing
        return {"X_inducing": X_inducing}

    def __get_bijectors__(self):
        return {"X_inducing": Identity()}

    def __get_priors__(self):
        return {"X_inducing": NoPrior()}
