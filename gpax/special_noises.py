import jax
import jax.numpy as jnp

from jaxtyping import Array
from gpax.kernels import RBFKernel

from gpax.noises import Noise
from gpax import ExactGP
from gpax.bijectors import Identity, Exp
from gpax.distributions import Zero, Normal


class HeteroscedasticNoise(Noise):
    def __init__(
        self,
        X_inducing=None,
        use_kernel_inducing=True,
        std_latent_noise=None,
        std_latent_noise_prior=Normal(),
        latent_gp_lengthscale_prior=Exp()(Zero()),
        latent_gp_variance_prior=Exp()(Zero()),
        train_latent_gp_noise=False,
    ):
        self.X_inducing = X_inducing
        self.use_kernel_inducing = use_kernel_inducing
        self.std_latent_noise = std_latent_noise
        self.std_latent_noise_prior = std_latent_noise_prior
        self.train_latent_gp_noise = train_latent_gp_noise
        self.noise_gp = ExactGP(
            RBFKernel(lengthscale_prior=latent_gp_lengthscale_prior, variance_prior=latent_gp_variance_prior)
        )

    def __call__(self, params, X):
        if self.X_inducing is not None:
            X_inducing = params["noise"]["X_inducing"]  # Use X_inducing from noise
        elif self.use_kernel_inducing:
            X_inducing = params["kernel"]["X_inducing"]  # Use X_inducing from kernel
        else:
            X_inducing = params["X_inducing"]  # Use X_inducing from a GP (SparseGP, etc.)

        if self.train_latent_gp_noise is False:
            params["noise"]["noise_gp"]["noise"]["variance"] = jax.lax.stop_gradient(
                params["noise"]["noise_gp"]["noise"]["variance"]
            )

        latent_cov = self.noise_gp.kernel(params["noise"]["noise_gp"])(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        # jax.debug.print(
        #     "latent_cov={latent_cov}, X_inducing={X_inducing}", latent_cov=latent_cov, X_inducing=X_inducing
        # )
        latent_log_noise = jnp.linalg.cholesky(latent_cov) @ params["noise"]["std_latent_noise"]
        params = params["noise"]

        return jnp.exp(
            self.noise_gp.predict(params["noise_gp"], X_inducing, latent_log_noise, X, return_cov=False)
        ).squeeze()  # squeeze is needed to make (n, 1) -> (n,)

    def __initialise_params__(self, key, X_inducing):
        priors = self.__get_priors__()
        params = {}
        if self.X_inducing is not None:
            X_inducing = self.X_inducing
            params["X_inducing"] = X_inducing

        key, subkey = jax.random.split(key)
        params["noise_gp"] = self.noise_gp.initialise_params(key, X_inducing)
        if self.train_latent_gp_noise is False:
            params["noise_gp"]["noise"]["variance"] = jnp.jitter

        if self.std_latent_noise is None:
            params["std_latent_noise"] = priors["std_latent_noise"].sample(subkey, (X_inducing.shape[0],))
        else:
            params["std_latent_noise"] = self.std_latent_noise
        return params

    def __get_bijectors__(self):
        bijectors = {"noise_gp": self.noise_gp.get_bijectors(), "std_latent_noise": Identity()}
        if self.X_inducing is not None:
            bijectors["X_inducing"] = Identity()
        return bijectors

    def __get_priors__(self):
        priors = {"noise_gp": self.noise_gp.get_priors(), "std_latent_noise": self.std_latent_noise_prior}
        if self.X_inducing is not None:
            priors["X_inducing"] = Zero()  # Dummy prior, never used for sampling
        return priors
