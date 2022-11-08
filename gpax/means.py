import jax.numpy as jnp
from gpax.core import Base, get_default_bijector


class Mean(Base):
    pass


class Scalar(Mean):
    def __init__(self, value=0.0, value_prior=None):
        self.value = value
        self.value_prior = value_prior

    def __call__(self, params, aux=None):
        return params["value"]

    def __initialize_params__(self, aux=None):
        params = {"value": self.value}
        self.constraints = {"value": get_default_bijector()}
        self.priors = {"value": self.value_prior}
        return params


class Zero(Mean):
    def __call__(self, params, aux=None):
        return aux["y"].mean()

    def __initialize_params__(self, aux=None):
        params = {}
        self.constraints = {}
        self.priors = {}
        return params
