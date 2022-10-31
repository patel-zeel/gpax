import jax.numpy as jnp
from numpy import var
from gpax.core import Base, get_default_prior, get_positive_bijector


class Likelihood(Base):
    """
    A meta class to define a likelihood.
    """

    pass


class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, variance_prior=None):
        self.variance = variance
        self.priors = {"variance": variance_prior}
        self.constraints = {"variance": get_positive_bijector()}

    def __call__(self, params, penalty=None):
        if penalty is not None:
            return params["variance"], jnp.array(0.0)
        else:
            return params["variance"]

    def __initialize_params__(self, aux):
        return {"variance": self.variance}


# class HeteroscedasticLikelihood(Likelihood):
#     def __init__(
#         self,
#         X_inducing=None,
#         use_kernel_inducing=True,
#         std_latent_noise=None,
#         std_latent_noise_prior=None,
#         latent_gp_lengthscale_prior=None,
#         latent_gp_variance_prior=None,
#         train_latent_gp_noise=False,
#         non_centered=True,
#     ):
#         self.X_inducing = X_inducing
#         self.use_kernel_inducing = use_kernel_inducing
#         self.std_latent_noise = std_latent_noise
#         self.std_latent_noise_prior = std_latent_noise_prior
#         self.train_latent_gp_noise = train_latent_gp_noise
#         self.non_centered = non_centered
#         self.noise_gp = ExactGP(
#             RBFKernel(lengthscale_prior=latent_gp_lengthscale_prior, variance_prior=latent_gp_variance_prior)
#         )

#     def get_posterior_penalty(self, params, X):
#         _, pred_cov = self(params, X, return_cov=True)
#         chol = jnp.linalg.cholesky(pred_cov)
#         return jnp.sum(jnp.log(jnp.diagonal(chol)))

#     def __call__(self, params, X, return_cov=False):
#         if self.X_inducing is not None:
#             X_inducing = params["noise"]["X_inducing"]  # Use X_inducing from noise
#         elif self.use_kernel_inducing:
#             X_inducing = params["kernel"]["X_inducing"]  # Use X_inducing from kernel
#         else:
#             X_inducing = params["X_inducing"]  # Use X_inducing from a GP (SparseGP, etc.)

#         if self.train_latent_gp_noise is False:
#             params["noise"]["noise_gp"]["noise"]["variance"] = jax.lax.stop_gradient(
#                 params["noise"]["noise_gp"]["noise"]["variance"]
#             )

#         if self.non_centered:
#             latent_cov = self.noise_gp.kernel(params["noise"]["noise_gp"])(X_inducing, X_inducing)
#             latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
#                 jnp.jitter + params["noise"]["noise_gp"]["noise"]["variance"]
#             )
#             # jax.debug.print(
#             #     "latent_cov={latent_cov}, X_inducing={X_inducing}", latent_cov=latent_cov, X_inducing=X_inducing
#             # )
#             latent_log_noise = jnp.linalg.cholesky(latent_cov) @ params["noise"]["std_latent_noise"]
#         else:
#             latent_log_noise = params["noise"]["std_latent_noise"]
#         params = params["noise"]

#         if return_cov:
#             pred_mean, pred_cov = self.noise_gp.predict(
#                 params["noise_gp"], X_inducing, latent_log_noise, X, return_cov=return_cov
#             )
#             return jnp.exp(pred_mean).squeeze(), pred_cov
#         else:
#             return jnp.exp(
#                 self.noise_gp.predict(params["noise_gp"], X_inducing, latent_log_noise, X, return_cov=return_cov)
#             ).squeeze()  # squeeze is needed to make (n, 1) -> (n,)

#     def __initialize_params__(self, key, X_inducing):
#         priors = self.__get_priors__()
#         params = {}
#         if self.X_inducing is not None:
#             X_inducing = self.X_inducing
#             params["X_inducing"] = X_inducing

#         key, subkey = jax.random.split(key)
#         params["noise_gp"] = self.noise_gp.initialize_params(key, X_inducing)
#         if self.train_latent_gp_noise is False:
#             params["noise_gp"]["noise"]["variance"] = jnp.array(0.0)

#         if self.std_latent_noise is None:
#             params["std_latent_noise"] = priors["std_latent_noise"].sample(subkey, (X_inducing.shape[0],))
#         else:
#             params["std_latent_noise"] = self.std_latent_noise
#         return params

#     def __get_bijectors__(self):
#         bijectors = {"noise_gp": self.noise_gp.get_bijectors(), "std_latent_noise": Identity()}
#         if self.X_inducing is not None:
#             bijectors["X_inducing"] = Identity()
#         return bijectors

#     def __get_priors__(self):
#         priors = {"noise_gp": self.noise_gp.get_priors(), "std_latent_noise": self.std_latent_noise_prior}
#         if self.X_inducing is not None:
#             priors["X_inducing"] = None
#         return priors


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

#     def __get_bijectors__(self):
#         bijectors = {"std_latent_noise_std": Identity()}

#         return bijectors

#     def __get_priors__(self):
#         priors = {"std_latent_noise_std": self.std_latent_noise_prior}
#         return priors
