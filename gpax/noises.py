import jax.numpy as jnp
import jax.tree_util as tree_util

from gpax.base import Base
from gpax.bijectors import Exp


class Noise(Base):
    def __call__(self, params, X):
        return self.call(params, X)

    def call(self, params, X):
        raise NotImplementedError("Must be implemented by subclass")

    def initialise_params(self, key, X_inducing=None):
        if self.__class__.__name__ == "HeteroscedasticNoise":
            params = {"noise": self.__initialise_params__(key, X_inducing=X_inducing)}
        else:
            params = {"noise": self.__initialise_params__(key)}
        return tree_util.tree_map(lambda x: jnp.asarray(x), params)

    def get_bijectors(self):
        return {"noise": self.__get_bijectors__()}


class HomoscedasticNoise(Noise):
    def __init__(self, variance=1.0):
        self.variance = variance

    def call(self, params, X):
        return params["noise"]["variance"]

    def __initialise_params__(self, key):
        return {"variance": self.variance}

    def __get_bijectors__(self):
        return {"variance": Exp()}
