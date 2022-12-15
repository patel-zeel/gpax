from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from gpax.core import Parameter, get_positive_bijector
from jaxtyping import Array, Float

from gpax.core import Module
from gpax.models import LatentGPHeinonen, LatentGPDeltaInducing
from gpax.kernels import RBF


class Likelihood(Module):
    """
    A meta class to define a likelihood.
    """

    pass


class Gaussian(Likelihood):
    def __init__(self, scale: Float[Array, "1"] = 1.0):
        super(Gaussian, self).__init__()
        self.scale = Parameter(scale, get_positive_bijector())

    def get_likelihood_fn(self, X_inducing: Parameter = None):
        return self.likelihood_fn

    def likelihood_fn(self, X):
        if self.training:
            return self.scale.get_value(), 0.0
        else:
            return self.scale.get_value()


class Heteroscedastic(Likelihood):
    def __init__(
        self,
        X_inducing: Array,
        latent_lengthscale: Float[Array, "1"] = 1.0,
        latent_scale: Float[Array, "1"] = 1.0,
        latent_kernel_type: type = RBF,
    ):
        super(Heteroscedastic, self).__init__()
        self.latent_gp = self.latent_gp_type(
            X=X_inducing,
            lengthscale=latent_lengthscale,
            scale=latent_scale,
            latent_kernel_type=latent_kernel_type,
            vmap=False,
        )

    def get_likelihood_fn(self, X_inducing: Parameter = None):
        if isinstance(self, HeteroscedasticHeinonen):
            X_inducing = jax.lax.stop_gradient(X_inducing())
        else:
            X_inducing = X_inducing()

        def likelihood_fn(X, X_new=None):
            return self.latent_gp(X_inducing, X, X_new)

        return likelihood_fn


class HeteroscedasticHeinonen(Heteroscedastic):
    def __init__(
        self,
        X_inducing: Array,
        latent_lengthscale: Float[Array, "1"] = 1.0,
        latent_scale: Float[Array, "1"] = 1.0,
        latent_kernel_type: type = RBF,
    ):
        self.latent_gp_type = LatentGPHeinonen
        super(HeteroscedasticHeinonen, self).__init__(X_inducing, latent_lengthscale, latent_scale, latent_kernel_type)


class HeteroscedasticDeltaInducing(Heteroscedastic):
    def __init__(
        self,
        X_inducing: Array,
        latent_lengthscale: Float[Array, "1"] = 1.0,
        latent_scale: Float[Array, "1"] = 1.0,
        latent_kernel_type: type = RBF,
    ):
        self.latent_gp_type = LatentGPDeltaInducing
        super(HeteroscedasticDeltaInducing, self).__init__(
            X_inducing, latent_lengthscale, latent_scale, latent_kernel_type
        )
