import jax
import jax.numpy as jnp

from jaxtyping import Array
from gpax.kernels import RBFKernel

from gpax.noises import Noise
from gpax import ExactGP
from gpax.bijectors import Identity, Exp
from gpax.distributions import Zero


class HeteroscedasticNoise(Noise):
    def __init__(
        self,
        X_inducing=None,
        X_inducing_prior=Zero(),
        use_kernel_inducing=True,
        latent_log_noise=None,
        latent_lengthscale_prior=Zero(),
        latent_variance_prior=Zero(),
    ):
        self.X_inducing = X_inducing
        self.use_kernel_inducing = use_kernel_inducing
        self.latent_log_noise = latent_log_noise
        self.noise_gp = ExactGP(
            RBFKernel(lengthscale_prior=latent_lengthscale_prior, variance_prior=latent_variance_prior)
        )
        if X_inducing is not None:
            self.X_inducing_prior = X_inducing_prior

    def __call__(self, params, X):
        if self.X_inducing is not None:
            X_inducing = params["noise"]["X_inducing"]  # Use X_inducing from noise
        elif self.use_kernel_inducing:
            X_inducing = params["kernel"]["X_inducing"]  # Use X_inducing from kernel
        else:
            X_inducing = params["X_inducing"]  # Use X_inducing from a GP (SparseGP, etc.)

        latent_cov = self.noise_gp.kernel(params["noise"]["noise_gp"])(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        # jax.debug.print(
        #     "latent_cov={latent_cov}, X_inducing={X_inducing}", latent_cov=latent_cov, X_inducing=X_inducing
        # )
        latent_log_noise = jnp.linalg.cholesky(latent_cov) @ params["noise"]["latent_log_noise"]
        params = params["noise"]
        return jnp.exp(
            self.noise_gp.predict(params["noise_gp"], X_inducing, latent_log_noise, X, return_cov=False)
        ).squeeze()  # squeeze is needed to make (n, 1) -> (n,)

    def __initialise_params__(self, key, X_inducing):
        params = {}
        if self.X_inducing is not None:
            X_inducing = self.X_inducing
            params["X_inducing"] = X_inducing

        params["noise_gp"] = self.noise_gp.initialise_params(key, X_inducing)

        if self.latent_log_noise is None:
            params["latent_log_noise"] = jnp.zeros(X_inducing.shape[0])
        else:
            params["latent_log_noise"] = self.latent_log_noise
        return params

    def __get_bijectors__(self):
        bijectors = {"noise_gp": self.noise_gp.get_bijectors(), "latent_log_noise": Identity()}
        if self.X_inducing is not None:
            bijectors["X_inducing"] = Identity()
        return bijectors

    def __get_priors__(self):
        priors = {"noise_gp": self.noise_gp.get_priors(), "latent_log_noise": Zero()}
        if self.X_inducing is not None:
            priors["X_inducing"] = self.X_inducing_prior
        return priors
