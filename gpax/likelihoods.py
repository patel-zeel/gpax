from __future__ import annotations

from gpax.core import Parameter
from chex import dataclass
from typing import Union
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
    scale: Union[Parameter, float] = 1.0

    def __post_init__(self):
        if not isinstance(self.scale, Parameter):
            self.scale = Parameter(self.scale, gb.get_positive_bijector())

    def __call__(self, X, train_mode=True):  # train_mode is for api consistency
        return self.scale()

    def __get_params__(self):
        return {"scale": self.scale}

    def set_params(self, params):
        self.scale.set(params["scale"])


@dataclass
class HeteroscedasticGaussian(Likelihood):
    scale_inducing: Union[Parameter, Array] = 1.0
    method: str = "gp_neurips"  # "heinonen"
    latent_gp: Model = None
    X_inducing: Parameter = None

    def __post_init__(self):
        if not isinstance(self.scale_inducing, Parameter):
            assert self.X_inducing is not None, "X_inducing must be provided if scale_inducing is not a Parameter."
            assert self.latent_gp is not None, "latent_gp must be provided if scale_inducing is not a Parameter."
            bijector = gb.InverseWhite(latent_gp=self.latent_gp, X_inducing=self.X_inducing)
            prior = bijector(gd.Normal(loc=0.0, scale=1.0))
            inversed_prior = gb.get_positive_bijector()(gd.Normal(loc=0.0, scale=1.0))
            self.scale_inducing = Parameter(
                self.scale_inducing, bijector, prior, inversed_init=True, inversed_prior=inversed_prior
            )

    def __get_params__(self):
        params = {
            "latent_gp": self.scale_inducing._bijector.latent_gp.__get_params__(),
            "scale_inducing": self.scale_inducing,
        }
        params["X_inducing"] = self.scale_inducing._bijector.X_inducing
        return params

    def set_params(self, raw_params):
        self.scale_inducing._bijector.latent_gp.set_params(raw_params["latent_gp"])
        self.scale_inducing.set(raw_params["scale_inducing"])
        self.scale_inducing._bijector.X_inducing.set(raw_params["X_inducing"])

    def __call__(self, X: Array, train_mode: bool = True):
        positive_bijector = gb.get_positive_bijector()
        X_inducing = self.scale_inducing._bijector.X_inducing()
        scale_inducing = self.scale_inducing()

        if self.method == "gp_neurips":  # Does not depend on train_mode
            train_mode = False
        elif self.method == "heinonen":
            pass
        else:
            raise ValueError(f"{self.method=} is not implemented.")

        if train_mode:
            return scale_inducing
        else:
            raw_scale_inducing = positive_bijector.inverse(scale_inducing)
            raw_scale = self.scale_inducing._bijector.latent_gp.predict(
                X_inducing, raw_scale_inducing, X, include_noise=False, return_cov=False, include_train_likelihood=False
            )
            return positive_bijector(raw_scale)


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
