from functools import partial
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

from jaxtyping import Array
from gpax.models import ExactGP
from gpax.kernels import Kernel, RBFKernel
from gpax.bijectors import Identity, Exp
from gpax.likelihoodshoods import HomoscedasticNoise
from gpax.means import ScalarMean, ZeroMean


class GibbsKernel(Kernel):
    def __init__(
        self,
        base_kernel="rbf",
        X_inducing=None,
        flex_scale=True,
        flex_variance=True,
        active_dims=None,
        ARD=True,
        scale_scale_prior=None,
        scale_sigma_prior=None,
        std_latent_scale_prior=None,
        std_latent_variance_prior=None,
        variance_scale_prior=None,
        variance_sigma_prior=None,
        lengthscale_prior=None,
        variance_prior=None,
        train_latent_gp_noise=False,
        non_centered=True,
    ):
        super().__init__(active_dims=active_dims, ARD=ARD)
        self.base_kernel = base_kernel
        self.X_inducing = X_inducing
        self.flex_scale = flex_scale
        self.flex_variance = flex_variance
        self.train_latent_gp_noise = train_latent_gp_noise
        self.non_centered = non_centered

        if self.flex_scale:
            self.std_latent_scale_prior = std_latent_scale_prior
            self.scale_gp = ExactGP(RBFKernel(lengthscale_prior=scale_scale_prior, variance_prior=scale_sigma_prior))
        else:
            self.lengthscale_prior = lengthscale_prior

        if self.flex_variance:
            self.std_latent_variance_prior = std_latent_variance_prior
            self.variance_gp = ExactGP(
                RBFKernel(lengthscale_prior=variance_scale_prior, variance_prior=variance_sigma_prior)
            )
        else:
            self.variance_prior = variance_prior

    def get_posterior_penalty(self, params, X):
        if not self.flex_scale and not self.flex_variance:
            return jnp.array(0.0)
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]

        penalty = jnp.array(0.0)
        if self.flex_scale:
            if self.train_latent_gp_noise is False:
                params["kernel"]["scale_gp"]["noise"]["variance"] = jax.lax.stop_gradient(
                    params["kernel"]["scale_gp"]["noise"]["variance"]
                )
            f = partial(
                self.predict_scale_per_dim,
                return_cov=True,
            )
            f = jax.vmap(f, in_axes=(1, 1, 0, 1))
            _, pred_cov = f(X, X_inducing, params["kernel"]["scale_gp"], params["kernel"]["std_latent_scale"])
            penalty += jax.vmap(lambda cov: jnp.log(jnp.diag(jnp.linalg.cholesky(cov))).sum())(pred_cov).sum()
        if self.flex_variance:
            if self.train_latent_gp_noise is False:
                params["kernel"]["variance_gp"]["noise"]["variance"] = jax.lax.stop_gradient(
                    params["kernel"]["variance_gp"]["noise"]["variance"]
                )

            _, pred_cov = self.predict_variance(params, X, return_cov=True)
            chol = jnp.linalg.cholesky(pred_cov)
            penalty += jnp.log(chol.diagonal()).sum()
        return penalty

    def call(self, params):
        def kernel_fn(x1, x2):
            if self.flex_scale:
                predict_fn = tree_util.Partial(self.predict_scale, params=params)
                l1 = predict_fn(x=x1)
                l2 = predict_fn(x=x2)
                l_avg_square = (l1**2 + l2**2) / 2.0
                prefix_part = jnp.sqrt(l1 * l2 / l_avg_square).prod()

                lengthscale = jnp.sqrt(l_avg_square)
            else:
                prefix_part = 1.0
                lengthscale = params["kernel"]["lengthscale"]

            sqr_dist = ((x1 / lengthscale - x2 / lengthscale) ** 2).sum()
            if self.base_kernel == "rbf":
                suffix_part = jnp.exp(-0.5 * sqr_dist)
            elif self.base_kernel == "matern12":
                suffix_part = jnp.exp(-jnp.sqrt(sqr_dist))
            elif self.base_kernel == "matern32":
                arg = jnp.sqrt(3.0) * jnp.sqrt(sqr_dist)
                suffix_part = (1.0 + arg) * jnp.exp(-arg)
            elif self.base_kernel == "matern52":
                arg = jnp.sqrt(5.0) * jnp.sqrt(sqr_dist)
                suffix_part = (1.0 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)

            if self.flex_variance:
                predict_fn = tree_util.Partial(self.predict_variance, params=params)
                var1 = predict_fn(x=x1.reshape(1, -1))
                var2 = predict_fn(x=x2.reshape(1, -1))
                variance_part = var1 * var2
            else:
                variance_part = params["kernel"]["variance"]

            return (variance_part * prefix_part * suffix_part).squeeze()

        return kernel_fn

    def predict_scale_per_dim(self, x, X_inducing, scale_gp_params, std_latent_scale, return_cov=False):
        x = x.reshape(-1, 1)
        X_inducing = X_inducing.reshape(-1, 1)
        if self.non_centered:
            latent_cov = self.scale_gp.kernel(scale_gp_params)(X_inducing, X_inducing)
            latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (jnp.jitter + scale_gp_params["noise"]["variance"])
            latent_Log_scale = jnp.linalg.cholesky(latent_cov) @ std_latent_scale
        else:
            latent_Log_scale = std_latent_scale
        if return_cov:
            pred_mean, pred_cov = self.scale_gp.predict(
                scale_gp_params, X_inducing, latent_Log_scale, x, return_cov=return_cov
            )
            return jnp.exp(pred_mean).squeeze(), pred_cov
        else:
            return jnp.exp(
                self.scale_gp.predict(scale_gp_params, X_inducing, latent_Log_scale, x, return_cov=return_cov)
            ).squeeze()

    def predict_scale(self, params, x):
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]

        if self.train_latent_gp_noise is False:
            params["kernel"]["scale_gp"]["noise"]["variance"] = jax.lax.stop_gradient(
                params["kernel"]["scale_gp"]["noise"]["variance"]
            )

        params = params["kernel"]

        f = jax.vmap(self.predict_scale_per_dim, in_axes=(None, 1, 0, 1))
        return f(x, X_inducing, params["scale_gp"], params["std_latent_scale"])

    def predict_variance(self, params, x, return_cov=False):
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]
        params = params["kernel"]

        if self.train_latent_gp_noise is False:
            params["variance_gp"]["noise"]["variance"] = jax.lax.stop_gradient(
                params["variance_gp"]["noise"]["variance"]
            )

        if self.non_centered:
            latent_cov = self.variance_gp.kernel(params["variance_gp"])(X_inducing, X_inducing)
            latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
                jnp.jitter + params["variance_gp"]["noise"]["variance"]
            )
            latent_Log_variance = jnp.linalg.cholesky(latent_cov) @ params["std_latent_variance"]
        else:
            latent_Log_variance = params["std_latent_variance"]
        if return_cov:
            pred_mean, pred_cov = self.variance_gp.predict(
                params["variance_gp"], X_inducing, latent_Log_variance, x, return_cov=return_cov
            )
            return jnp.exp(pred_mean).squeeze(), pred_cov
        else:
            return jnp.exp(
                self.variance_gp.predict(
                    params["variance_gp"],
                    X_inducing,
                    latent_Log_variance,
                    x,
                    return_cov=return_cov,
                )
            ).squeeze()

    def __initialize_params__(self, key, X_inducing=None):
        params = {}
        priors = self.__get_priors__()
        key, subkey = jax.random.split(key, 2)
        if self.flex_scale:
            if self.X_inducing is not None:  # Ignore X_inducing if self.X_inducing is given
                X_inducing = self.X_inducing
                params["X_inducing"] = X_inducing
            else:
                assert X_inducing is not None, "X_inducing must not be None if self.X_inducing is None"

            def initialize_per_dim(key, x_inducing):
                params = self.scale_gp.initialize_params(key, x_inducing.reshape(-1, 1))
                if self.train_latent_gp_noise is False:
                    params["noise"]["variance"] = jnp.array(0.0)
                return params

            keys = jax.random.split(key, X_inducing.shape[1] + 1)
            params["scale_gp"] = jax.vmap(initialize_per_dim, in_axes=(0, 1))(keys[:-1], X_inducing)

            params["std_latent_scale"] = priors["std_latent_scale"].sample(keys[-1], sample_shape=X_inducing.shape)
        else:
            params["lengthscale"] = priors["lengthscale"].sample(key)
        if self.flex_variance:
            keys = jax.random.split(subkey, 2)
            if self.X_inducing is not None:
                X_inducing = self.X_inducing
                params["X_inducing"] = X_inducing
            params["variance_gp"] = self.variance_gp.initialize_params(keys[0], X_inducing)

            if self.train_latent_gp_noise is False:
                params["variance_gp"]["noise"]["variance"] = jnp.array(0.0)

            params["std_latent_variance"] = priors["std_latent_variance"].sample(
                seed=keys[1], sample_shape=(X_inducing.shape[0],)
            )
        else:
            params["variance"] = priors["variance"].sample(subkey)
        return params

    def __get_bijectors__(self):
        bijectors = {}
        if self.X_inducing is not None:
            bijectors["X_inducing"] = tfb.Identity()
        if self.flex_scale:
            if self.X_inducing is not None:
                bijectors["X_inducing"] = Identity()
            bijectors["scale_gp"] = self.scale_gp.get_bijectors()
            bijectors["std_latent_scale"] = Identity()
        else:
            bijectors["lengthscale"] = Exp()
        if self.flex_variance:
            if self.X_inducing is not None:
                bijectors["X_inducing"] = Identity()
            bijectors["variance_gp"] = self.variance_gp.get_bijectors()
            bijectors["std_latent_variance"] = Identity()
        else:
            bijectors["variance"] = Exp()
        return bijectors

    def __get_priors__(self):
        priors = {}
        if self.flex_scale:
            if self.X_inducing is not None:
                priors["X_inducing"] = None  # Dummy prior, never used for sampling
            priors["scale_gp"] = self.scale_gp.get_priors()
            priors["std_latent_scale"] = self.std_latent_scale_prior
        else:
            priors["lengthscale"] = self.lengthscale_prior
        if self.flex_variance:
            if self.X_inducing is not None:
                priors["X_inducing"] = None  # Dummy prior, never used for sampling
            priors["variance_gp"] = self.variance_gp.get_priors()
            priors["std_latent_variance"] = self.std_latent_variance_prior
        else:
            priors["variance"] = self.variance_prior
        return priors


class HeinonenGibbsKernel(Kernel):
    def __init__(
        self,
        flex_scale=True,
        flex_variance=True,
        active_dims=None,
        ARD=True,
        std_latent_scale_prior=None,
        std_latent_variance_prior=None,
        lengthscale_prior=None,
        variance_prior=None,
        train_latent_gp_noise=False,
        scale_gp_lengthscale=0.1,
        scale_gp_variance=1.0,
        scale_gp_noise=0.0,
        variance_gp_lengthscale=0.1,
        variance_gp_variance=1.0,
        variance_gp_noise=0.0,
    ):
        super().__init__(active_dims=active_dims, ARD=ARD)
        self.flex_scale = flex_scale
        self.flex_variance = flex_variance
        self.train_latent_gp_noise = train_latent_gp_noise

        if self.flex_scale:
            self.std_latent_scale_prior = std_latent_scale_prior
            self.scale_gp = ExactGP(
                RBFKernel(lengthscale=scale_gp_lengthscale, variance=scale_gp_variance),
                noise=HomoscedasticNoise(variance=scale_gp_noise),
                mean=ZeroMean(),
            )
            self.scale_gp_params = self.scale_gp.initialize_params(jax.random.PRNGKey(0), jnp.array([[0.0]]))
        else:
            self.lengthscale_prior = lengthscale_prior

        if self.flex_variance:
            self.std_latent_variance_prior = std_latent_variance_prior
            self.variance_gp = ExactGP(
                RBFKernel(lengthscale=variance_gp_lengthscale, variance=variance_gp_variance),
                noise=HomoscedasticNoise(variance=variance_gp_noise),
                mean=ZeroMean(),
            )
            self.variance_gp_params = self.variance_gp.initialize_params(jax.random.PRNGKey(0), jnp.array([[0.0]]))
        else:
            self.variance_prior = variance_prior

    def train_latent_log_variance_std(self, params):
        X_inducing = params["X_inducing"]
        latent_cov = self.variance_gp.kernel(self.variance_gp_params)(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
            jnp.jitter + self.variance_gp_params["noise"]["variance"]
        )
        return jnp.linalg.cholesky(latent_cov) @ params["kernel"]["std_latent_variance_std"]

    def train_latent_log_lengthscale(self, params):
        X_inducing = params["X_inducing"]
        latent_cov = self.scale_gp.kernel(self.scale_gp_params)(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
            jnp.jitter + self.scale_gp_params["noise"]["variance"]
        )
        return jnp.linalg.cholesky(latent_cov) @ params["kernel"]["std_latent_scale"]

    def train_cov(self, params, return_prior_log_prob=False):
        X_inducing = params["X_inducing"]

        sqr_dist = (
            (X_inducing**2).sum(axis=1, keepdims=1)
            + (X_inducing**2).T.sum(axis=0, keepdims=1)
            - 2.0 * X_inducing @ X_inducing.T
        )
        prior_log_prob = 0.0
        if self.flex_scale:
            latent_Log_scale = self.train_latent_log_lengthscale(params)
            prior_log_prob += self.scale_gp.log_probability(self.scale_gp_params, X_inducing, latent_Log_scale)
            latent_scale = jnp.exp(latent_Log_scale).squeeze()

            l_avg_square = (latent_scale.reshape(1, -1) ** 2 + latent_scale.reshape(-1, 1) ** 2) / 2.0
            prefix_part = jnp.sqrt(latent_scale.reshape(1, -1) * latent_scale.reshape(-1, 1) / l_avg_square)

            exp_part = jnp.exp(-sqr_dist / (2.0 * l_avg_square))

        else:
            prefix_part = 1.0
            exp_part = jnp.exp(-sqr_dist / (2.0 * params["kernel"]["lengthscale"] ** 2))

        if self.flex_variance:
            latent_Log_variance_std = self.train_latent_log_variance_std(params)
            prior_log_prob += self.variance_gp.log_probability(
                self.variance_gp_params, X_inducing, latent_Log_variance_std
            )
            latent_variance_std = jnp.exp(latent_Log_variance_std).squeeze()

            variance_part = latent_variance_std.reshape(1, -1) * latent_variance_std.reshape(-1, 1)
        else:
            variance_part = params["kernel"]["variance"]

        cov = prefix_part * exp_part * variance_part
        if return_prior_log_prob:
            return cov, prior_log_prob
        else:
            return cov

    def call(self, params):
        def kernel_fn(x1, x2):
            if self.flex_scale:
                predict_fn = tree_util.Partial(self.predict_scale, params=params)
                l1 = predict_fn(x=x1.reshape(1, -1))
                l2 = predict_fn(x=x2.reshape(1, -1))
                l_avg_square = (l1**2 + l2**2) / 2.0
                l_avg = jnp.sqrt(l_avg_square)
                prefix_part = jnp.sqrt(l1 * l2 / l_avg_square).prod()
            else:
                l_avg = params["kernel"]["lengthscale"]
                prefix_part = jnp.array(1.0)

            sqr_dist = ((x1 / l_avg - x2 / l_avg) ** 2).sum()
            exp_part = jnp.exp(-0.5 * sqr_dist)

            if self.flex_variance:
                predict_fn = tree_util.Partial(self.predict_variance_std, params=params)
                var_std1 = predict_fn(x=x1.reshape(1, -1))
                var_std2 = predict_fn(x=x2.reshape(1, -1))
                variance_part = var_std1 * var_std2
            else:
                variance_part = params["kernel"]["variance"]

            cov = (variance_part * prefix_part * exp_part).squeeze()
            return cov

        return kernel_fn

    def predict_scale(self, params, x):
        X_inducing = params["X_inducing"]
        params = params["kernel"]

        latent_cov = self.scale_gp.kernel(self.scale_gp_params)(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
            jnp.jitter + self.scale_gp_params["noise"]["variance"]
        )
        latent_Log_scale = jnp.linalg.cholesky(latent_cov) @ params["std_latent_scale"]

        return jnp.exp(self.scale_gp.predict(self.scale_gp_params, X_inducing, latent_Log_scale, x, return_cov=False))

    def predict_variance_std(self, params, x):
        X_inducing = params["X_inducing"]
        params = params["kernel"]

        latent_cov = self.variance_gp.kernel(self.variance_gp_params)(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * (
            jnp.jitter + self.variance_gp_params["noise"]["variance"]
        )
        latent_Log_variance_std = jnp.linalg.cholesky(latent_cov) @ params["std_latent_variance_std"]

        pred_log_variance_std = self.variance_gp.predict(
            self.variance_gp_params, X_inducing, latent_Log_variance_std, x, return_cov=False
        )

        return jnp.exp(pred_log_variance_std)

    def __initialize_params__(self, key, X_inducing=None):
        params = {}
        priors = self.__get_priors__()
        key, subkey = jax.random.split(key, 2)
        if self.flex_scale:
            key = jax.random.split(key, 1)[0]
            params["std_latent_scale"] = priors["std_latent_scale"].sample(key, sample_shape=(X_inducing.shape[0],))
        else:
            params["lengthscale"] = priors["lengthscale"].sample(key)
        if self.flex_variance:
            key = jax.random.split(key, 1)[0]
            params["std_latent_variance_std"] = priors["std_latent_variance_std"].sample(
                seed=key, sample_shape=(X_inducing.shape[0],)
            )
        else:
            params["variance"] = priors["variance"].sample(subkey)
        return params

    def __get_bijectors__(self):
        bijectors = {}
        if self.flex_scale:
            bijectors["std_latent_scale"] = Identity()
        else:
            bijectors["lengthscale"] = Exp()
        if self.flex_variance:
            bijectors["std_latent_variance_std"] = Identity()
        else:
            bijectors["variance"] = Exp()
        return bijectors

    def __get_priors__(self):
        priors = {}
        if self.flex_scale:
            priors["std_latent_scale"] = self.std_latent_scale_prior
        else:
            priors["lengthscale"] = self.lengthscale_prior
        if self.flex_variance:
            priors["std_latent_variance_std"] = self.std_latent_variance_prior
        else:
            priors["variance"] = self.variance_prior
        return priors
