import jax.numpy as jnp
from gpax.core import Base, get_default_bijector


class Mean(Base):
    pass


class ScalarMean(Mean):
    def __init__(self, value=0.0, value_prior=None):
        self.value = value

        self.constraints = {"value": get_default_bijector()}
        self.priors = {"value": value_prior}

    def __call__(self, params, aux=None):
        return params["value"]

    def __initialize_params__(self, aux):
        return {"value": self.value}

    def __get_priors__(self):
        return


class ZeroMean(Mean):
    def __init__(self):
        self.constraints = {}
        self.priors = {}

    def __call__(self, params, aux):
        return aux["y"].mean()

    def __initialize_params__(self, aux):
        return {}
