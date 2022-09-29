import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

from jaxtyping import Array
from gpax.gps import ExactGP
from gpax.kernels import Kernel, RBFKernel
from gpax.bijectors import Identity, Exp
from gpax.distributions import Zero


class GibbsKernel(Kernel):
    def __init__(
        self,
        X_inducing=None,
        flex_scale=True,
        flex_variance=True,
        active_dims=None,
        ARD=True,
        X_inducing_prior=Zero(),
        scale_scale_prior=Zero(),
        scale_sigma_prior=Zero(),
        variance_scale_prior=Zero(),
        variance_sigma_prior=Zero(),
    ):
        super().__init__(active_dims=active_dims, ARD=ARD)
        self.X_inducing = X_inducing
        self.flex_scale = flex_scale
        self.flex_variance = flex_variance
        self.scale_scale_prior = scale_scale_prior
        self.scale_sigma_prior = scale_sigma_prior
        self.variance_scale_prior = variance_scale_prior
        self.variance_sigma_prior = variance_sigma_prior

        if self.X_inducing is not None:
            self.prior = {"X_inducing": X_inducing_prior}

        if self.flex_scale:
            self.scale_gp = ExactGP(RBFKernel(lengthscale_prior=scale_scale_prior, variance_prior=scale_sigma_prior))

        if self.flex_variance:
            self.variance_gp = ExactGP(
                RBFKernel(lengthscale_prior=variance_scale_prior, variance_prior=variance_sigma_prior)
            )

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

    def predict_scale_per_dim(self, x, X_inducing, scale_gp_params, latent_log_scale, params):
        x = x.reshape(1, -1)
        X_inducing = X_inducing.reshape(-1, 1)
        latent_cov = self.scale_gp.kernel(params["scale_gp"])(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        latent_log_scale = jnp.linalg.cholesky(latent_cov) @ latent_log_scale
        return jnp.exp(
            self.scale_gp.predict(scale_gp_params, X_inducing, latent_log_scale, x, return_cov=False)
        ).squeeze()

    def predict_scale(self, params, x):
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]
        params = params["kernel"]
        f = jax.vmap(self.predict_scale_per_dim, in_axes=(None, 1, 0, 1, None))
        return f(x, X_inducing, params["scale_gp"], params["latent_log_scale"], params)

    def predict_var(self, params, x):
        x = x.reshape(1, -1)
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]
        params = params["kernel"]
        variance_gp = ExactGP(kernel=RBFKernel(active_dims=list(range(X_inducing.shape[1]))))
        latent_cov = variance_gp.kernel(params["variance_gp"])(X_inducing, X_inducing)
        latent_cov = latent_cov + jnp.eye(X_inducing.shape[0]) * jnp.jitter
        latent_log_variance = jnp.linalg.cholesky(latent_cov) @ params["latent_log_variance"]
        res = jnp.exp(
            variance_gp.predict(
                params["variance_gp"],
                X_inducing,
                latent_log_variance,
                x,
                return_cov=False,
            )
        ).squeeze()
        # jax.debug.print("{res}", res=res)
        return res

    def __initialise_params__(self, key, X_inducing=None):
        params = {}
        if self.flex_scale:
            if self.X_inducing is not None:  # Ignore X_inducing if self.X_inducing is given
                X_inducing = self.X_inducing
                params["X_inducing"] = X_inducing
            else:
                assert X_inducing is not None, "X_inducing must not be None if self.X_inducing is None"

            keys = jax.random.split(key, X_inducing.shape[1])

            def initialize_per_dim(key, x_inducing):
                params = self.scale_gp.initialise_params(key, x_inducing.reshape(-1, 1))
                return params

            params["scale_gp"] = jax.vmap(initialize_per_dim, in_axes=(0, 1))(keys, X_inducing)
            params["latent_log_scale"] = jnp.zeros(X_inducing.shape)
        else:
            params["lengthscale"] = jnp.array(1.0)
        if self.flex_variance:
            if self.X_inducing is not None:
                X_inducing = self.X_inducing
                params["X_inducing"] = X_inducing
            key = jax.random.split(key, 1)[0]
            params["variance_gp"] = self.variance_gp.initialise_params(key, X_inducing)
            params["latent_log_variance"] = jnp.zeros(X_inducing.shape[0])
        else:
            params["variance"] = jnp.array(1.0)
        return params

    def __get_bijectors__(self):
        bijectors = {}
        if self.flex_scale:
            if self.X_inducing is not None:
                bijectors["X_inducing"] = Identity()
            bijectors["scale_gp"] = self.scale_gp.get_bijectors()
            bijectors["latent_log_scale"] = Identity()
        else:
            bijectors["lengthscale"] = Exp()
        if self.flex_variance:
            if self.X_inducing is not None:
                bijectors["X_inducing"] = Identity()
            bijectors["variance_gp"] = self.variance_gp.get_bijectors()
            bijectors["latent_log_variance"] = Identity()
        else:
            bijectors["variance"] = Exp()
        return bijectors

    def __get_priors__(self):
        priors = {}
        if self.flex_scale:
            if self.X_inducing is not None:
                priors["X_inducing"] = Zero()
            priors["scale_gp"] = self.scale_gp.get_priors()
            priors["latent_log_scale"] = Zero()
        else:
            priors["lengthscale"] = Zero()
        if self.flex_variance:
            if self.X_inducing is not None:
                priors["X_inducing"] = Zero()
            priors["variance_gp"] = self.variance_gp.get_priors()
            priors["latent_log_variance"] = Zero()
        else:
            priors["variance"] = Zero()
        return priors
