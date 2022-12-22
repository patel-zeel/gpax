import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util as jtu


import tensorflow_probability.substrates.jax as tfp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from jaxtyping import Array, PyTree

tfb = tfp.bijectors
tfd = tfp.distributions

DEFAULT_PRIOR = lambda: tfd.Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))


def get_default_prior():
    return DEFAULT_PRIOR()


def set_default_prior(prior):
    global DEFAULT_PRIOR
    DEFAULT_PRIOR = lambda: prior


DEFAULT_JITTER = 1e-6


def get_default_jitter():
    return DEFAULT_JITTER


def set_default_jitter(jitter):
    global DEFAULT_JITTER
    DEFAULT_JITTER = jitter


POSITIVE_BIJECTOR = tfb.Softplus()


def get_positive_bijector():
    return POSITIVE_BIJECTOR


def set_positive_bijector(bijector):
    global POSITIVE_BIJECTOR
    POSITIVE_BIJECTOR = bijector


is_parameter = lambda x: isinstance(x, Parameter)


class Parameter:
    def __init__(
        self,
        value: Union[float, Array],
        bijector: tfb.Bijector = None,
        prior: tfd.Distribution = None,
        trainable: bool = True,
        fixed_init: bool = False,
    ):
        self.bijector = bijector if bijector is not None else tfb.Identity()
        self.prior = prior
        self._trainable = trainable
        self.fixed_init = fixed_init  # if True, the value is not changed during initialization
        self._shape = jnp.asarray(value).shape
        self._raw_value = self.bijector.inverse(jnp.asarray(value))
        self._raw_shape = self._raw_value.shape

    def get_value(self):
        if self._trainable is False:
            self._raw_value = jax.lax.stop_gradient(self._raw_value)
        return self.bijector(self._raw_value)

    def trainable(self, is_trainable: bool = True):
        self._trainable = is_trainable
        return self

    def __call__(self):  # for convenience
        return self.get_value()

    def get_raw_value(self):
        return self._raw_value

    def set_raw_value(self, raw_value):
        assert raw_value.shape == self._raw_shape
        self._raw_value = jnp.array(raw_value)

    def set_value(self, value):
        assert jnp.asarray(value).shape == self._shape
        self._raw_value = self.bijector.inverse(jnp.asarray(value))

    def initialize(self, key):
        if self.fixed_init or self._trainable is False:
            return
        elif self.prior is None:
            self._raw_value = get_default_prior().sample(self._raw_value.shape, key)
        else:
            value = self.prior.sample(self._shape, key)
            self._raw_value = self.bijector.inverse(value)

    def log_prior(self):
        if self.prior is None or (self.trainable is False):
            return jnp.zeros_like(self._raw_value)
        return self.prior.log_prob(self.bijector(self._raw_value))


class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def train(self):
        self.training = True
        for module in self.modules():
            module.train()
        return self

    def eval(self):
        self.training = False
        for module in self.modules():
            module.eval()
        return self

    def trainable(self, is_trainable: bool = True):
        for param in self._parameters.values():
            param.trainable(is_trainable)
        for module in self.modules():
            module.trainable(is_trainable)
        return self

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self._parameters:
            return self._parameters[key]

        if key in self._modules:
            return self._modules[key]

    def modules(self):
        return self._modules.values()

    def get_raw_parameters(self, raw_dict=True):
        params = jtu.tree_map(lambda x: x, self._parameters)  # copy
        for module_name, module in self._modules.items():
            params[module_name] = module.get_raw_parameters(raw_dict=False)

        if raw_dict:
            return jtu.tree_map(lambda x: x.get_raw_value(), params, is_leaf=is_parameter)
        else:
            return params

    def get_parameters(self, raw_dict=True):
        params = self.get_raw_parameters(raw_dict=False)

        if raw_dict:
            return jtu.tree_map(lambda x: x(), params, is_leaf=is_parameter)
        else:
            return params

    def set_raw_parameters(self, raw_params):
        params = self.get_raw_parameters(raw_dict=False)
        jtu.tree_map(lambda param, value: param.set_raw_value(value), params, raw_params, is_leaf=is_parameter)

    def set_parameters(self, params):
        raw_params = self.get_parameters(raw_dict=False)
        jtu.tree_map(lambda param, value: param.set_value(value), raw_params, params, is_leaf=is_parameter)

    def initialize(self, key):
        param_objects = self.get_raw_parameters(raw_dict=False)
        flat_params, _ = jtu.tree_flatten(param_objects, is_leaf=is_parameter)
        seeds = [seed for seed in jax.random.split(key, len(flat_params))]
        jtu.tree_map(lambda seed, param: param.initialize(seed), seeds, flat_params)
        return self.get_raw_parameters(raw_dict=True)

    def log_prior(self):
        params = self.raw_parameters(raw_dict=False)
        log_prior_values = jtu.tree_map(lambda x: x.log_prior(), params, is_leaf=is_parameter)
        return ravel_pytree(log_prior_values)[0].sum()
