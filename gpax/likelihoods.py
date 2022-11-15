import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from gpax.core import (
    Model,
    pytree,
    Likelihood,
    get_default_prior,
    get_positive_bijector,
    get_default_bijector,
    get_default_jitter,
)
from gpax.utils import add_to_diagonal, repeat_to_size
from chex import dataclass
from dataclasses import field
from jaxtyping import Array

import gpax.distributions as gd


@pytree
@dataclass
class Gaussian(Likelihood):
    variance: float = 1.0
    variance_prior: gd.Distribution = None

    def __post_init__(self):
        self.priors = {"variance": self.variance_prior}
        self.constraints = {"variance": get_positive_bijector()}

    def __call__(self):
        return self.variance

    def get_params(self):
        return {"variance": self.variance}

    def set_params(self, params):
        self.variance = params["variance"]

    def tree_flatten(self):
        aux = {"variance_prior": self.variance_prior, "constraints": self.constraints}
        return (self.variance,), aux

    @classmethod
    def tree_unflatten(cls, params, aux):
        return cls(variance=params[0], **aux)


@pytree
@dataclass
class HeteroscedasticGaussian(Likelihood):
    latent_gp: Model = None
    X_inducing: Array = None
    variance: float = 1.0
    whitened_raw_variance: Array = None
    prior_type: str = "gp_neurips"

    def __post_init__(self):
        assert self.latent_gp is not None, "latent_gp must be provided"
        assert self.X_inducing is not None, "X_inducing must be provided"
        self.modules = {"latent_gp": self.latent_gp}
        self.priors = {
            "latent_gp": self.latent_gp.priors,
            "variance": None,
        }
        self.constraints = {
            "latent_gp": self.latent_gp.constraints,
            "variance": get_positive_bijector(),
        }

    def get_params(self):
        params = {"latent_gp": self.latent_gp.get_params()}
        if "variance" in self.priors:
            params["variance"] = self.variance
        else:
            params["whitened_raw_variance"] = self.whitened_raw_variance
        return params

    def set_params(self, params):
        self.latent_gp.set_params(params["latent_gp"])
        if "variance" in self.priors:
            self.variance = params["variance"]
        else:
            self.whitened_raw_variance = params["whitened_raw_variance"]

    def _post_init_params__(self):
        if self.variance is None:
            return
        self.priors["whitened_raw_variance"] = get_default_prior()
        self.constraints["whitened_raw_variance"] = get_default_bijector()
        self.priors.pop("variance")
        self.constraints.pop("variance")

        positive_bijector = get_positive_bijector()

        variance = repeat_to_size(variance, self.X_inducing.shape[0])

        raw_covariance = self.latent_gp.kernel(self.X_inducing, self.X_inducing)
        raw_noise = 0.0
        noisy_raw_covariance = add_to_diagonal(raw_covariance, raw_noise, get_default_jitter())
        cholesky = jnp.linalg.cholesky(noisy_raw_covariance)
        raw_variance = positive_bijector.inverse(variance)
        raw_mean = self.latent_gp.mean()
        self.whitened_raw_variance = jsp.linalg.solve_triangular(cholesky, raw_variance - raw_mean, lower=True)

    def __call__(self, X):
        positive_bijector = get_positive_bijector()
        if self.prior_type == "gp_neurips":
            inducing_cov = self.latent_gp.kernel(self.X_inducing, self.X_inducing)
            stable_inducing_cov = add_to_diagonal(inducing_cov, 0.0, get_default_jitter())
            cholesky = jnp.linalg.cholesky(stable_inducing_cov)

            raw_mean = self.latent_gp.mean()

            raw_variance = raw_mean + cholesky @ self.whitened_raw_variance

            raw_infered_variance = self.latent_gp.predict(self.X_inducing, raw_variance, X)
            infered_variance = positive_bijector(raw_infered_variance)
            return infered_variance
        else:
            raise NotImplementedError(f"{self.prior_type=} is not implemented.")

    def tree_flatten(self):
        latent_gp_params, latent_gp_treedef = jtu.tree_flatten(self.latent_gp)

    @classmethod
    def tree_unflatten(cls, params, aux):
        aux[""]
        return cls(variance=params[0], **aux)


# class HeinonenHeteroscedasticNoise(Noise):
#     def __init__(
#         self,
#         std_latent_noise=None,
#         std_latent_noise_prior=None,
#         latent_gp_lengthscale_prior=None,
#         latent_gp_variance_prior=None,
#         noise_gp_lengthscale=0.1,
#         noise_gp_variance=1.0,
#         noise_gp_noise=0,
#     ):
#         self.std_latent_noise = std_latent_noise
#         self.std_latent_noise_prior = std_latent_noise_prior
#         self.noise_gp = ExactGP(
#             kernel=RBFKernel(lengthscale=noise_gp_lengthscale, variance=noise_gp_variance),
#             noise=HomoscedasticNoise(variance=noise_gp_noise),
#             mean=ZeroMean(),
#         )

#         self.noise_gp_params = self.noise_gp.initialize_params(jax.random.PRNGKey(0), jnp.array([[0.0]]))

#     def train_noise(self, params, return_prior_log_prob=False):
#         X_inducing = params["X_inducing"]
#         params = params["noise"]
#         latent_cov = self.noise_gp.kernel(self.noise_gp_params)(X_inducing, X_inducing)
#         latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
#             jnp.jitter + self.noise_gp_params["noise"]["variance"]
#         )
#         latent_log_noise_std = jnp.linalg.cholesky(latent_cov) @ params["std_latent_noise_std"]
#         latent_noise = jnp.exp(latent_log_noise_std) ** 2
#         if return_prior_log_prob:
#             prior_log_prob = self.noise_gp.log_probability(self.noise_gp_params, X_inducing, latent_log_noise_std)
#             return latent_noise, prior_log_prob

#         return latent_noise

#     def call(self, params, X):
#         X_inducing = params["X_inducing"]
#         latent_cov = self.noise_gp.kernel(self.noise_gp_params)(X_inducing, X_inducing)
#         latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
#             jnp.jitter + self.noise_gp_params["noise"]["variance"]
#         )
#         latent_log_noise_std = jnp.linalg.cholesky(latent_cov) @ params["noise"]["std_latent_noise_std"]

#         pred_log_noise_std = self.noise_gp.predict(
#             self.noise_gp_params, X_inducing, latent_log_noise_std, X, return_cov=False
#         )
#         return jnp.exp(pred_log_noise_std).squeeze() ** 2

#     def __initialize_params__(self, key, X_inducing):
#         priors = self.__get_priors__()
#         params = {}

#         key, subkey = jax.random.split(key)

#         if self.std_latent_noise is None:
#             params["std_latent_noise_std"] = priors["std_latent_noise_std"].sample(subkey, (X_inducing.shape[0],))
#         else:
#             params["std_latent_noise_std"] = self.std_latent_noise
#         return params
