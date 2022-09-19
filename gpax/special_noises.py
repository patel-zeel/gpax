import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors

from jaxtyping import Array

from chex import dataclass
from gpax.noises import Noise
from gpax import ExactGP

import lab.jax as B


@dataclass
class HeteroscedasticNoise(Noise):
    inducing_std_noise: Array = None
    X_inducing: Array = None
    use_kernel_inducing: bool = True

    def __post_init__(self):
        self.noise_gp = ExactGP()

    def __call__(self, params, X):
        if self.X_inducing is not None:
            X_inducing = params["noise"]["X_inducing"]  # Use X_inducing from noise
        elif self.use_kernel_inducing:
            X_inducing = params["kernel"]["X_inducing"]  # Use X_inducing from kernel
        else:
            X_inducing = params["X_inducing"]  # Use X_inducing from a GP (SparseGP, etc.)

        params = params["noise"]

        covar = B.dense(self.noise_gp.kernel(params["noise_gp"])(X_inducing, X_inducing))
        covar += jnp.eye(covar.shape[0]) * params["noise_gp"]["noise"]["variance"]
        latent_log_variance = (
            params["noise_gp"]["mean"]["value"] + jnp.linalg.cholesky(covar) @ params["inducing_std_noise"]
        )

        return jnp.exp(
            self.noise_gp.predict(params["noise_gp"], X_inducing, latent_log_variance, X, return_cov=False)
        ).squeeze()  # squeeze is needed to make (n, 1) -> (n,)

    def __initialise_params__(self, key, X_inducing):
        params = {}
        if self.X_inducing is not None:
            X_inducing = self.X_inducing
            params["X_inducing"] = X_inducing

        params["noise_gp"] = self.noise_gp.initialise_params(key, X_inducing)

        if self.inducing_std_noise is None:
            params["inducing_std_noise"] = jnp.zeros(X_inducing.shape[0])
        else:
            params["inducing_std_noise"] = self.inducing_std_noise
        return params

    def __get_bijectors__(self):
        bijectors = {"noise_gp": self.noise_gp.get_bijectors(), "inducing_std_noise": tfb.Identity()}
        if self.X_inducing is not None:
            bijectors["X_inducing"] = tfb.Identity()
        return bijectors
