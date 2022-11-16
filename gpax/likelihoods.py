from __future__ import annotations
from gpax.core import Parameter
from chex import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from jaxtyping import Array

import gpax.distributions as gd
import gpax.bijectors as gb

from gpax.core import Module


@dataclass
class Likelihood(Module):
    """
    A meta class to define a likelihood.
    """

    pass


@dataclass
class Gaussian(Likelihood):
    scale: float = 1.0
    scale_prior: gd.Distribution = None

    def __post_init__(self):
        self.scale = Parameter(self.scale, gb.get_positive_bijector(), self.scale_prior)

    def __call__(self, X):
        return self.scale()

    def __get_params__(self):
        return {"scale": self.scale}

    def set_params(self, params):
        self.scale.set(params["scale"])


@dataclass
class HeteroscedasticGaussian(Likelihood):
    latent_gp: Model = None
    X_inducing: Array = None
    scale: float = 1.0
    scale_prior: gd.Distribution = field(default_factory=lambda: gb.get_positive_bijector()(gd.Normal()))
    type: str = "gp_neurips"

    def __post_init__(self):
        assert self.latent_gp is not None, "latent_gp must be provided."
        self.common_params = {}  # placeholder for common parameters
        if self.X_inducing is not None:
            self.X_inducing = Parameter(self.X_inducing, gb.get_default_bijector(), gd.Fixed())

        white_bijector = gb.White(mean=self.latent_gp.mean, kernel_fn=self.latent_gp.kernel)
        self.scale = Parameter(self.scale, white_bijector, self.scale_prior)

    def get_X_inducing(self):
        if "X_inducing" in self.common_params:
            return self.common_params["X_inducing"]
        else:
            assert isinstance(self.X_inducing, Parameter)
            return self.X_inducing

    def __get_params__(self):
        params = {"latent_gp": self.latent_gp.__get_params__(), "scale": self.scale}
        if "X_inducing" not in self.common_params and self.X_inducing is not None:
            params["X_inducing"] = self.X_inducing

        # This is a hack to assign X_inducing to the bijector. Not a good practice in general.
        self.scale._bijector.X_inducing = self.get_X_inducing()

        return params

    def set_params(self, params):
        self.latent_gp.set_params(params["latent_gp"])
        self.scale.set(params["scale"])
        if self.X_inducing is not None:
            self.X_inducing.set(params["X_inducing"])

    def __call__(self, X):
        if self.type == "gp_neurips":
            X_inducing = self.get_X_inducing()()
            inducing_scale = self.scale()
            scale_X = self.latent_gp.predict(X_inducing, inducing_scale, X, include_noise=False, return_cov=False)
            return scale_X
        else:
            raise NotImplementedError(f"{self.prior_type=} is not implemented.")


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
