import jax.numpy as jnp
import jax.tree_util as tree_util

from gpax.base import Base
from gpax.bijectors import Exp
from gpax.distributions import Zero
from gpax.utils import get_raw_log_prior


class Noise(Base):
    def __call__(self, params, X):
        return self.call(params, X)

    def log_prior(self, params, bijectors):
        params = params["noise"]
        bijectors = bijectors["noise"]
        return {"noise": get_raw_log_prior(self.prior, params, bijectors)}

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

    def get_priors(self):
        return {"noise": self.__get_priors__()}


class HomoscedasticNoise(Noise):
    def __init__(self, variance=None, variance_prior=Exp()(Zero())):
        self.variance = variance
        self.variance_prior = variance_prior

    def call(self, params, X):
        return params["noise"]["variance"]

    def __initialise_params__(self, key):
        priors = self.__get_priors__()
        if self.variance is not None:
            return {"variance": self.variance}
        else:
            return {"variance": priors["variance"].sample(key)}

    def __get_bijectors__(self):
        return {"variance": Exp()}

    def __get_priors__(self):
        return {"variance": self.variance_prior}
