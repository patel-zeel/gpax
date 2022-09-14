import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors

from jaxtyping import Array
from chex import dataclass
from gpax.stheno.kernels import GibbsKernel as GibbsKernelStheno
from gpax.gps import ExactGP
from gpax.kernels import Kernel

import lab.jax as B


@dataclass
class GibbsKernel(Kernel):
    X_inducing: Array = None
    flex_scale: bool = True
    flex_variance: bool = True

    def __post_init__(self):
        if self.X_inducing is not None:
            self.X_inducing = B.uprank(self.X_inducing)

    def call(self, params):
        if self.X_inducing is not None:
            X_inducing = params["kernel"]["X_inducing"]
        else:
            X_inducing = params["X_inducing"]
        kernel = GibbsKernelStheno(
            X_inducing=X_inducing, flex_scale=self.flex_scale, flex_variance=self.flex_variance, params=params["kernel"]
        )
        return kernel

    def predict_scale(self, params, x):
        return self.call(params).predict_scale(x)

    def predict_var(self, params, x):
        return self.call(params).predict_var(x)

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
                return ExactGP().initialise_params(key, x_inducing)

            params["scale_gp"] = jax.vmap(initialize_per_dim, in_axes=(0, 1))(keys, X_inducing)
            params["latent_log_scale"] = jnp.zeros(X_inducing.shape)
        else:
            params["lengthscale"] = jnp.array(1.0)
        if self.flex_variance:
            if self.X_inducing is not None:
                X_inducing = self.X_inducing
                params["X_inducing"] = X_inducing
            key = jax.random.split(key, 1)[0]
            params["variance_gp"] = ExactGP().initialise_params(key, X_inducing)
            params["latent_log_variance"] = jnp.zeros(X_inducing.shape[0])
        else:
            params["variance"] = jnp.array(1.0)
        return params

    def __get_bijectors__(self):
        bijectors = {}
        if self.flex_scale:
            if self.X_inducing is not None:
                bijectors["X_inducing"] = tfb.Identity()
            bijectors["scale_gp"] = ExactGP().get_bijectors()
            bijectors["latent_log_scale"] = tfb.Identity()
        else:
            bijectors["lengthscale"] = tfb.Exp()
        if self.flex_variance:
            if self.X_inducing is not None:
                bijectors["X_inducing"] = tfb.Identity()
            bijectors["variance_gp"] = ExactGP().get_bijectors()
            bijectors["latent_log_variance"] = tfb.Identity()
        else:
            bijectors["variance"] = tfb.Exp()
        return bijectors
