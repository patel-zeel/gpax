import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from gpax.utils import constrain, unconstrain

from gpax.core import Base, get_default_jitter, get_default_bijector
from gpax.utils import add_to_diagonal, get_a_inv_b


class Model(Base):
    def constrain(self, params):
        return constrain(params, self.constraints)

    def unconstrain(self, params):
        return unconstrain(params, self.constraints)


class GPRegression(Model):
    def __init__(self, kernel, likelihood, mean, n_dims, n_inducing=None):
        self.components = {"kernel": kernel, "likelihood": likelihood, "mean": mean}
        self.n_dims = n_dims
        self.n_inducing = n_inducing

    def __initialize_params__(self, aux):
        params = jtu.tree_map(lambda x: x.__initialize_params__(aux), self.components)

        # constraints
        self.constraints = {
            "kernel": self.kernel.constraints,
            "likelihood": self.likelihood.constraints,
            "mean": self.mean.constraints,
        }

        if self.X_inducing is not None:
            self.constraints["X_inducing"] = get_default_bijector()

        # priors
        self.priors = {
            "kernel": self.kernel.priors,
            "likelihood": self.likelihood.priors,
            "mean": self.mean.priors,
        }

        if self.X_inducing is not None:
            self.priors["X_inducing"] = None

        return params

    def __post_initialize_params__(self, params, aux):
        return {
            "kernel": self.kernel.__post_initialize_params__(params["kernel"], aux),
            "likelihood": self.likelihood.__post_initialize_params__(params["likelihood"], aux),
            "mean": self.mean.__post_initialize_params__(params["mean"], aux),
        }

    def log_probability(self, params, X, y):
        """
        prior_type: default: None, possible values: "prior", "posterior", None
        """
        # kernel
        covariance = self.kernel(params["kernel"])(X, X)

        # likelihood
        likelihood_variance = self.likelihood(params["likelihood"], X, X_inducing)

        # mean
        mean = self.mean(params["mean"], aux={"y": y})

        y_bar = y - mean
        noisy_covariance = add_to_diagonal(covariance, likelihood_variance, get_default_jitter())

        k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)

        log_likelihood = (
            -0.5 * (y_bar.ravel() * k_inv_y.ravel()).sum()  # This will break for multi-dimensional y
            - jnp.log(k_cholesky.diagonal()).sum()
            - 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)
        )

        if prior_type is None:
            return log_likelihood
        else:
            log_prior = log_likelihood_prior + log_kernel_prior
            return log_likelihood + log_prior

    def condition(self, params, X, y):
        """
        This function is useful while doing batch prediction.
        """

        # kernel
        kernel_fn = self.kernel(params["kernel"])
        train_cov = kernel_fn(X, X)

        # likelihood
        noise_variance = self.likelihood(params["likelihood"])

        # mean
        mean = self.mean(params["mean"], aux={"y": y})

        y_bar = y - mean
        noisy_covariance = add_to_diagonal(train_cov, noise_variance, get_default_jitter())
        k_inv_y, k_cholesky = get_a_inv_b(noisy_covariance, y_bar, return_cholesky=True)

        def predict_fn(X_test, return_cov=True, include_noise=True):
            K_star = kernel_fn(X_test, X)
            pred_mean = K_star @ k_inv_y + mean

            if return_cov:
                k_inv_k_star = jsp.linalg.cho_solve((k_cholesky, True), K_star.T)
                pred_cov = kernel_fn(X_test, X_test) - (K_star @ k_inv_k_star)
                if include_noise:
                    aux = {"X": X_test}
                    if "X_inducing" in params:
                        aux["X_inducing"] = params["X_inducing"]
                    pred_cov = add_to_diagonal(
                        pred_cov, self.likelihood(params["likelihood"], aux=aux), get_default_jitter()
                    )
                return pred_mean, pred_cov
            else:
                return pred_mean

        return predict_fn

    def predict(self, params, X, y, X_test, return_cov=True, include_noise=True):
        """
        This method is suitable for one time prediction.
        In case of batch prediction, it is better to use `condition` method in combination with `predict`.
        """
        predict_fn = self.condition(params, X, y)
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
