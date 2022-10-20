import jax.numpy as jnp
import jax.tree_util as tree_util

from gpax.base import Base
from gpax.bijectors import Identity
from gpax.distributions import NoPrior


class Mean(Base):
    def __call__(self, params):
        return self.call(params["mean"])

    def log_prior(self):
        return {"mean": self.__get_priors__()}

    def initialise_params(self, key):
        params = {"mean": self.__initialise_params__(key)}
        return tree_util.tree_map(lambda x: jnp.asarray(x), params)

    def get_bijectors(self):
        return {"mean": self.__get_bijectors__()}

    def get_priors(self):
        return {"mean": self.__get_priors__()}


class ScalarMean(Mean):
    def __init__(self, value=None, value_prior=NoPrior()):
        self.value = value
        self.value_prior = value_prior

    def call(self, params):
        return params["value"]

    def log_prior(self, params):
        params = params["mean"]

    def __initialise_params__(self, key):
        priors = self.__get_priors__()
        if self.value is not None:
            return {"value": self.value}
        else:
            return {"value": priors["value"].sample(key)}

    def __get_bijectors__(self):
        return {"value": Identity()}

    def __get_priors__(self):
        return {"value": self.value_prior}


class ZeroMean(Mean):
    def call(self, params):
        return jnp.array(0.0)

    def log_prior(self, params):
        params = params["mean"]

    def __initialise_params__(self, key):
        return {}

    def __get_bijectors__(self):
        return {}

    def __get_priors__(self):
        return {}
