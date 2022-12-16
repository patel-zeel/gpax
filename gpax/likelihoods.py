from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from gpax.core import Parameter, get_positive_bijector
from jaxtyping import Array, Float

from gpax.core import Module, get_default_jitter
from gpax.models import LatentGPHeinonen, LatentGPDeltaInducing
from gpax.kernels import RBF
from gpax.means import Average
from gpax.utils import add_to_diagonal


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


class HeteroscedasticHeinonen(Likelihood):
    def __init__(
        self,
        X_inducing: Array,
        latent_gp_lengthscale: Float[Array, "1"] = 1.0,
        latent_gp_scale: Float[Array, "1"] = 1.0,
        latent_kernel_type: type = RBF,
    ):
        super(HeteroscedasticHeinonen, self).__init__()
        self.latent_gp = LatentGPHeinonen(
            X_inducing, latent_gp_lengthscale, latent_gp_scale, latent_kernel_type, vmap=False
        )

    def get_likelihood_fn(self, X_inducing: Array = None):
        return self.latent_gp(X_inducing)


class HeteroscedasticDeltaInducing(Likelihood):
    def __init__(
        self,
        X_inducing: Array,
        latent_gp_lengthscale: Float[Array, "1"] = 1.0,
        latent_gp_scale: Float[Array, "1"] = 1.0,
        latent_kernel_type: type = RBF,
    ):
        super(HeteroscedasticDeltaInducing, self).__init__()
        self.latent_gp = LatentGPDeltaInducing(
            X_inducing, latent_gp_lengthscale, latent_gp_scale, latent_kernel_type, vmap=False
        )

    def get_likelihood_fn(self, X_inducing: Array = None):
        return self.latent_gp(X_inducing)
