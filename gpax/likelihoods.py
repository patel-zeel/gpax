from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from gpax.core import Parameter, get_positive_bijector
from jaxtyping import Array, Float

from gpax.core import Module, get_default_jitter
from gpax.models import LatentModel
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
        if self._training:
            return self.scale.get_value().repeat(X.shape[0]), 0.0
        else:
            return self.scale.get_value().repeat(X.shape[0])

    def __repr__(self) -> str:
        return f"Gaussian"


class Heteroscedastic(Likelihood):
    def __init__(
        self,
        latent_model: LatentModel,
    ):
        super(Heteroscedastic, self).__init__()
        self.latent_model = latent_model

    def get_likelihood_fn(self, X_inducing: Array = None):
        return self.latent_model(X_inducing)

    def __repr__(self) -> str:
        return f"Heteroscedastic"
