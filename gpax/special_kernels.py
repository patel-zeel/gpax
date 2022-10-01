import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

from jaxtyping import Array
from gpax.gps import ExactGP
from gpax.kernels import Kernel, RBFKernel
from gpax.bijectors import Identity, Exp
from gpax.distributions import Zero, Normal


class GibbsKernel(Kernel):
    def __init__(
        self,
        X_inducing=None,
        flex_scale=True,
        flex_variance=True,
        active_dims=None,
        ARD=True,
        scale_scale_prior=Exp()(Zero()),
        scale_sigma_prior=Exp()(Zero()),
        std_latent_scale_prior=Normal(),
        std_latent_variance_prior=Normal(),
        variance_scale_prior=Exp()(Zero()),
        variance_sigma_prior=Exp()(Zero()),
        lengthscale_prior=Exp()(Zero()),
        variance_prior=Exp()(Zero()),
        train_latent_gp_noise=False,
    ):
        super().__init__(active_dims=active_dims, ARD=ARD)
        self.X_inducing = X_inducing
        self.flex_scale = flex_scale
        self.flex_variance = flex_variance
        self.train_latent_gp_noise = train_latent_gp_noise

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

    def call(self, params):
        def kernel_fn(x1, x2):
            if self.flex_scale:
                predict_fn = jax.jit(tree_util.Partial(self.predict_scale, params=params))
                l1 = predict_fn(x=x1)
                l2 = predict_fn(x=x2)
                l_avg_square = (l1**2 + l2**2) / 2.0
                l_avg = jnp.sqrt(l_avg_square)
                prefix_part = jnp.sqrt(l1 * l2 / l_avg_square).prod()
                x1 = x1 / l_avg
                x2 = x2 / l_avg
            else:
                lengthscale = params["kernel"]["lengthscale"]
                x1 = x1 / lengthscale
                x2 = x2 / lengthscale
                prefix_part = 1.0

            sqr_dist = ((x1 - x2) ** 2).sum()
            exp_part = jnp.exp(-0.5 * sqr_dist)

            if self.flex_variance:
                predict_fn = jax.jit(tree_util.Partial(self.predict_var, params=params))
                var1 = predict_fn(x=x1)
                var2 = predict_fn(x=x2)
                variance_part = var1 * var2
            else:
                variance_part = params["kernel"]["variance"]

            return (variance_part * prefix_part * exp_part).squeeze()

        return kernel_fn

    def predict_scale_per_dim(self, x, X_inducing, scale_gp_params, std_latent_scale, params):
        x = x.reshape(1, -1)
        X_inducing = X_inducing.reshape(-1, 1)
        latent_cov = self.scale_gp.kernel(params["scale_gp"])(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        latent_Log_scale = jnp.linalg.cholesky(latent_cov) @ std_latent_scale
        return jnp.exp(
            self.scale_gp.predict(scale_gp_params, X_inducing, latent_Log_scale, x, return_cov=False)
        ).squeeze()

    def predict_scale(self, params, x):
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]
        params = params["kernel"]

        if self.train_latent_gp_noise is False:
            params["scale_gp"]["noise"]["variance"] = jax.lax.stop_gradient(params["scale_gp"]["noise"]["variance"])

        f = jax.vmap(self.predict_scale_per_dim, in_axes=(None, 1, 0, 1, None))
        return f(x, X_inducing, params["scale_gp"], params["std_latent_scale"], params)

    def predict_var(self, params, x):
        x = x.reshape(1, -1)
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]
        params = params["kernel"]

        if self.train_latent_gp_noise is False:
            params["variance_gp"]["noise"]["variance"] = jax.lax.stop_gradient(
                params["variance_gp"]["noise"]["variance"]
            )

        latent_cov = self.variance_gp.kernel(params["variance_gp"])(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        latent_Log_variance = jnp.linalg.cholesky(latent_cov) @ params["std_latent_variance"]
        res = jnp.exp(
            self.variance_gp.predict(
                params["variance_gp"],
                X_inducing,
                latent_Log_variance,
                x,
                return_cov=False,
            )
        ).squeeze()
        # jax.debug.print("{res}", res=res)
        return res

    def __initialise_params__(self, key, X_inducing=None):
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
                params = self.scale_gp.initialise_params(key, x_inducing.reshape(-1, 1))
                if self.train_latent_gp_noise is False:
                    params["noise"]["variance"] = jnp.jitter
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
            params["variance_gp"] = self.variance_gp.initialise_params(keys[0], X_inducing)

            if self.train_latent_gp_noise is False:
                params["variance_gp"]["noise"]["variance"] = jnp.jitter

            params["std_latent_variance"] = priors["std_latent_variance"].sample(
                seed=keys[1], sample_shape=(X_inducing.shape[0],)
            )
        else:
            params["variance"] = priors["variance"].sample(subkey)
        return params

    def __get_bijectors__(self):
        bijectors = {}
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
                priors["X_inducing"] = Zero()  # Dummy prior, never used for sampling
            priors["scale_gp"] = self.scale_gp.get_priors()
            priors["std_latent_scale"] = self.std_latent_scale_prior
        else:
            priors["lengthscale"] = self.lengthscale_prior
        if self.flex_variance:
            if self.X_inducing is not None:
                priors["X_inducing"] = Zero()  # Dummy prior, never used for sampling
            priors["variance_gp"] = self.variance_gp.get_priors()
            priors["std_latent_variance"] = self.std_latent_variance_prior
        else:
            priors["variance"] = self.variance_prior
        return priors
