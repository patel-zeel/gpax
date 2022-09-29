import jax.numpy as jnp
import jax.tree_util as tree_util
from regex import D

from gpax.base import Base
from gpax.bijectors import Identity
from gpax.distributions import Zero
from gpax.utils import get_raw_log_prior


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


class ConstantMean(Mean):
    def __init__(self, value=0.0, value_prior=Zero()):
        self.value = value
        self.value_prior = value_prior

    def call(self, params):
        return params["value"]

    def log_prior(self, params):
        params = params["mean"]

    def __initialise_params__(self, key):
        if self.value is not None:
            return {"value": self.value}
        else:
            return {"value": jnp.array(0.0)}

    def __get_bijectors__(self):
        return {"value": Identity()}

    def __get_priors__(self):
        return {"value": self.value_prior}
