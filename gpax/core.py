import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util as jtu
import gpax.bijectors as gb
import gpax.distributions as gd
from chex import dataclass

from typing import Any

is_parameter = lambda x: isinstance(x, Parameter)

# Core classes
class Parameter:
    def __init__(
        self,
        value: Any,
        bijector: gb.Bijector = gb.get_default_bijector(),
        prior: gd.Distribution = None,
        fixed_init=False,
        inversed_init=False,
        inversed_prior=None,
    ):
        self.fixed_init = fixed_init
        self.inversed_init = inversed_init
        self.inversed_prior = inversed_prior
        self._bijector = bijector
        self.__raw_value = self._bijector.inverse(jnp.asarray(value))
        self._raw_prior = self._bijector.inverse(prior)
        self.__value_is_changed = True
        self.__buffer_value = None

    def __call__(self):
        if self.__value_is_changed:
            self.__buffer_value = self._bijector(self.__raw_value)
            self.__value_is_changed = False
        return self.__buffer_value

    def get(self):
        return self.__raw_value

    def set(self, raw_value):
        self.__raw_value = jnp.asarray(raw_value)
        self.__value_is_changed = True

    def initialize(self, key):
        if self.fixed_init:
            return
        if self._raw_prior is None:
            raw_prior = gd.get_default_prior()
        else:
            raw_prior = self._raw_prior

        if self.inversed_init:  # special case
            transformed_value = self.inversed_prior.sample(key, ())
            self.__raw_value = self._bijector.inverse(transformed_value)
        else:
            self.__raw_value = raw_prior.sample(key, self.__raw_value.shape)
        self.__value_is_changed = True

    def log_prior(self):
        if self._raw_prior is None:
            return jnp.zeros_like(self.__raw_value)
        return self._raw_prior.log_prob(self.__raw_value)


@dataclass
class Module:
    def get_params(self, raw_dict=True):
        params = self.__get_params__()
        if raw_dict:
            return jtu.tree_map(lambda param: param.get(), params, is_leaf=is_parameter)
        else:
            return params

    def get_constrained_params(self):
        params = self.get_params(raw_dict=False)
        return jtu.tree_map(lambda x: x(), params, is_leaf=is_parameter)

    def initialize(self, key):
        params = self.get_params(raw_dict=False)
        flat_params, _ = jtu.tree_flatten(params, is_leaf=is_parameter)
        seeds = [seed for seed in jax.random.split(key, len(flat_params))]
        jtu.tree_map(lambda seed, param: param.initialize(seed), seeds, flat_params)

    def log_prior(self):
        params = self.get_params(raw_dict=False)
        log_prior_values = jtu.tree_map(lambda x: x.log_prior(), params, is_leaf=is_parameter)
        return ravel_pytree(log_prior_values)[0].sum()
