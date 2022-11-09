import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from gpax.core import Base, get_default_prior, get_positive_bijector, get_default_bijector, get_default_jitter
from gpax.utils import add_to_diagonal, repeat_to_size


class Likelihood(Base):
    """
    A meta class to define a likelihood.
    """

    pass


class Gaussian(Likelihood):
    def __init__(self, variance=1.0, variance_prior=None):
        self.variance = variance
        self.variance_prior = variance_prior

    def __call__(self, params, prior_type=None):
        if prior_type is not None:
            return params["variance"], jnp.array(0.0)
        else:
            return params["variance"]

    def __initialize_params__(self, aux):
        params = {"variance": self.variance}
        self.priors = {"variance": self.variance_prior}
        self.constraints = {"variance": get_positive_bijector()}
        return params


class HeteroscedasticHeinonen(Likelihood):
    def __init__(
        self,
        latent_gp=None,
        likelihood_variance=1.0,
    ):
        self.latent_gp = latent_gp
        self.likelihood_variance = likelihood_variance

    def __initialize_params__(self, aux):
        params = {
            "latent_gp": self.latent_gp.__initialize_params__(aux),
            "dummy_likelihood_variance": self.likelihood_variance,
        }

        self.priors = {
            "latent_gp": self.latent_gp.priors,
            "dummy_likelihood_variance": None,
        }
        self.constraints = {
            "latent_gp": self.latent_gp.constraints,
            "dummy_likelihood_variance": get_positive_bijector(),
        }

        return params

    def __post_initialize_params__(self, params, aux):
        # Some initial jobs
        self.priors["whitened_raw_likelihood_variance"] = None
        self.constraints["whitened_raw_likelihood_variance"] = get_default_bijector()
        self.priors.pop("dummy_likelihood_variance")
        self.constraints.pop("dummy_likelihood_variance")

        likelihood_variance = params.pop("dummy_likelihood_variance")
        positive_bijector = get_positive_bijector()

        if "X_inducing" in aux:  # For inducing point methods
            X_prior = aux["X_inducing"]
        else:  
            X_prior = aux["X"]

        likelihood_variance = repeat_to_size(likelihood_variance, X_prior.shape[0])

        raw_covariance = self.latent_gp.kernel(params["latent_gp"]["kernel"])(X_prior, X_prior)
        raw_noise = 0.0
        noisy_raw_covariance = add_to_diagonal(raw_covariance, raw_noise, get_default_jitter())
        cholesky = jnp.linalg.cholesky(noisy_raw_covariance)
        raw_likelihood_variance = positive_bijector.inverse(likelihood_variance)
        whitened_raw_likelihood_variance = jsp.linalg.solve_triangular(cholesky, raw_likelihood_variance, lower=True)
        params["whitened_raw_likelihood_variance"] = whitened_raw_likelihood_variance
        return params

    def __call__(self, params, aux, prior_type=None):
        positive_bijector = get_positive_bijector()
        if "X_inducing" in aux:  # For inducing point methods
            latent_gp_mean = self.latent_gp.mean(params["latent_gp"]["mean"])  # Only scalar mean is supported
            latent_gp_mean = repeat_to_size(latent_gp_mean, aux["X_inducing"].shape[0])
            raw_covariance = self.latent_gp.kernel(params["latent_gp"]["kernel"])(aux["X_inducing"], aux["X_inducing"])
            raw_noise = 0.0
            noisy_covariance = add_to_diagonal(raw_covariance, raw_noise, get_default_jitter())
            cholesky = jnp.linalg.cholesky(noisy_covariance)
            cross_covariance = self.latent_gp.kernel(params["latent_gp"]["kernel"])(aux["X"], aux["X_inducing"])
            test_covariance = self.latent_gp.kernel(params["latent_gp"]["kernel"])(aux["X"], aux["X"])
            
            pred_mean = cross_covariance@jsp.linalg.cho_solve((cholesky, True), latent_gp_mean)
            k_inv_kt = jsp.linalg.cho_solve((cholesky, True), cross_covariance.T)
            pred_cov = test_covariance - cross_covariance@k_inv_kt + k_inv_kt.T@
            # raw_covariance = self.latent_gp.kernel(params["latent_gp"]["kernel"])(aux["X_inducing"], aux["X_inducing"])
            # raw_noise = self.latent_gp.likelihood(params["latent_gp"]["likelihood"])
            # noisy_raw_covariance = add_to_diagonal(raw_covariance, raw_noise, get_default_jitter())
            # raw_cholesky = jnp.linalg.cholesky(noisy_raw_covariance)
            # raw_likelihood_variance = raw_cholesky @ params["whitened_raw_likelihood_variance"]
        else:
            pass

        if "X_inducing" in aux:
            raw_inferred_likelihood_variance = self.latent_gp.predict(
                params["latent_gp"], X_prior, raw_likelihood_variance, aux["X"], return_cov=False
            )
            likelihood_variance = positive_bijector(raw_inferred_likelihood_variance)
        else:
            likelihood_variance = positive_bijector(raw_likelihood_variance)

        if prior_type is None:
            return likelihood_variance
        elif prior_type == "prior":  # Non-Stationary Gaussian Process Regression with Hamiltonian Monte Carlo
            # l ~ N(\mu, \Sigma)
            # LL^T = \Sigma
            # l_tilde ~ N(L^-1\mu, I)
            mu = self.latent_gp.mean(params["latent_gp"]["mean"], aux={"y": params["whitened_raw_likelihood_variance"]})
            mu = repeat_to_size(mu, X_prior.shape[0])
            whitened_mu = jsp.linalg.solve_triangular(raw_cholesky, mu, lower=True)
            log_prior = jsp.stats.norm.logpdf(
                params["whitened_raw_likelihood_variance"], loc=whitened_mu, scale=1.0
            ).sum()
            return likelihood_variance, log_prior
        elif prior_type == "posterior":  # Nonstationary Gaussian Process Regression Using Point Estimates of Local Smoothness
            


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
