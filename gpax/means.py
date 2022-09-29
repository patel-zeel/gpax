import jax.numpy as jnp
import jax.tree_util as tree_util

from gpax.base import Base
from gpax.bijectors import Identity


class Mean(Base):
    def initialise_params(self, key):
        params = {"mean": self.__initialise_params__(key)}
        return tree_util.tree_map(lambda x: jnp.asarray(x), params)

    def get_bijectors(self):
        return {"mean": self.__get_bijectors__()}


class ConstantMean(Mean):
    def __init__(self, value=0.0):
        self.value = value

    def call(self, params):
        params = params["mean"]
        return params["value"]

    def __initialise_params__(self, key):
        if self.value is not None:
            return {"value": self.value}
        else:
            return {"value": jnp.array(0.0)}

    def __get_bijectors__(self):
        return {"value": Identity()}
