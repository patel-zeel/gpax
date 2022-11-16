import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util as jtu
import gpax.bijectors as gb
import gpax.distributions as gd
from chex import dataclass

from typing import Any

from gpax.bijectors import get_default_bijector

is_parameter = lambda x: isinstance(x, Parameter)

# Core classes
class Parameter:
    def __init__(self, value: Any, bijector: gb.Bijector = None, prior: gd.Distribution = None):
        self._value = jnp.asarray(value)
        if bijector is None:
            self._bijector = get_default_bijector()
        else:
            self._bijector = bijector
        if prior is None:
            self._prior = self._bijector(gd.NoPrior())
        else:
            self._prior = prior

    def __call__(self):
        return self._value

    def set(self, value):
        self._value = jnp.asarray(value)

    def unconstrain(self):
        self._value = self._bijector.inverse(self._value)
        self._prior = self._bijector.inverse(self._prior)

    def constrain(self):
        self._value = self._bijector.forward(self._value)
        self._prior = self._bijector.forward(self._prior)

    def initialize(self, key):
        if isinstance(self._prior, gd.Fixed):
            pass
        else:
            self._value = self._prior.sample(key, self._value.shape)

    def log_prior(self):
        return self._prior.log_prob(self._value)


@dataclass
class Module:
    constrained: bool = True

    def get_params(self, raw_dict=True):
        params = self.__get_params__()
        if raw_dict:
            return jtu.tree_map(lambda x: x(), params, is_leaf=is_parameter)
        else:
            return params

    def constrain(self):
        if self.constrained is True:
            return
        params = self.get_params(raw_dict=False)
        jtu.tree_map(lambda param: param.constrain(), params, is_leaf=is_parameter)
        self.constrained = True

    def unconstrain(self):
        if self.constrained is False:
            return
        params = self.get_params(raw_dict=False)
        jtu.tree_map(lambda param: param.unconstrain(), params, is_leaf=is_parameter)
        self.constrained = False

    def initialize(self, key):
        params = self.get_params(raw_dict=False)
        flat_params, _ = jtu.tree_flatten(params, is_leaf=is_parameter)
        seeds = [seed for seed in jax.random.split(key, len(flat_params))]
        jtu.tree_map(lambda seed, param: param.initialize(seed), seeds, flat_params)

    def log_prior(self):
        params = self.get_params(raw_dict=False)
        log_prior_values = jtu.tree_map(lambda x: x.log_prior(), params, is_leaf=is_parameter)
        return ravel_pytree(log_prior_values)[0].sum()
