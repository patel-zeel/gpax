import jax.numpy as jnp
from gpax.core import pytree, Mean, get_default_bijector
import gpax.distributions as gd
from chex import dataclass
from dataclasses import field


@pytree
@dataclass
class Scalar(Mean):
    value: float = 1.0
    value_prior: gd.Distribution = None

    def __post_init__(self):
        self.constraints = {"value": get_default_bijector()}
        self.priors = {"value": self.value_prior}

    def __call__(self, aux=None):
        return self.value

    def get_params(self):
        return {"value": self.value}

    def set_params(self, params):
        self.value = params["value"]

    def tree_flatten(self):
        aux = {"value_prior": self.value_prior, "constraints": self.constraints}
        return (self.value,), aux

    @classmethod
    def tree_unflatten(cls, aux, params):
        return cls(value=params[0], **aux)


@pytree
@dataclass
class Average(Mean):
    def __post_init__(self):
        self.constraints = {}
        self.priors = {}

    def __call__(self, aux):
        return aux["y"].mean()

    def get_params(self):
        return {}

    def set_params(self, params):
        pass

    def tree_flatten(self):
        return (), {"constraints": self.constraints, "priors": self.priors}

    @classmethod
    def tree_unflatten(cls, aux, params):
        return cls(**aux)
