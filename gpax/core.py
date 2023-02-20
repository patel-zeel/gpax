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


# This condition is used to treat `Parameter` objects as leaves in a PyTree
is_parameter = lambda x: isinstance(x, Parameter)


@jtu.register_pytree_node_class
class Parameter:
    def __init__(
        self,
        value: Union[float, Array],
        bijector: tfb.Bijector = None,
        prior: tfd.Distribution = None,
        trainable: bool = True,  # if False, the value is not updated during training
        fixed_init: bool = False,  # if True, the value is not updated during initialization
    ):
        value = jnp.asarray(value)
        self.prior = prior
        self._trainable = trainable
        self._fixed_init = fixed_init

        # bijector
        if bijector is None:
            if self.prior is None:
                bijector = tfb.Identity()
            elif isinstance(self.prior, tfd.Distribution):
                bijector = self.prior._default_event_space_bijector()
            else:
                raise ValueError(f"prior must be a Distribution, not {type(self.prior)}")
        self.bijector = bijector

        # store value in unconstrained space
        self._raw_value = self.bijector.inverse(value)

        self._shape = value.shape
        self._raw_shape = self._raw_value.shape

    def get_value(self):
        if not self._trainable:
            raw_value = jax.lax.stop_gradient(self._raw_value)
        else:
            raw_value = self._raw_value
        return self.bijector(raw_value)

    def trainable(self, is_trainable: bool = True):
        self._trainable = is_trainable
        return self

    def __call__(self):  # for convenience
        return self.get_value()

    def get_raw_value(self):
        return self._raw_value

    def set_raw_value(self, raw_value):
        raw_value = jnp.asarray(raw_value)
        assert raw_value.shape == self._raw_shape, f"{raw_value.shape} != {self._raw_shape}"
        self._raw_value = raw_value

    def set_value(self, value):
        value = jnp.asarray(value)
        assert value.shape == self._shape, f"{value.shape} != {self._shape}"
        raw_value = self.bijector.inverse(value)
        self.set_raw_value(raw_value)

    def initialize(self, key):
        if self._trainable is False or self._fixed_init is True:
            return
        elif self.prior is None:
            default_prior = get_default_prior()
            raw_value = default_prior.sample(self._raw_value.shape, key)
            self.set_raw_value(raw_value)
        else:
            value = self.prior.sample(self._shape, key)
            self.set_value(value)

    def log_prob(self):
        if self.prior is None:
            return jnp.zeros_like(self._raw_value)
        else:
            return self.prior.log_prob(self.get_value())

    def __repr__(self) -> str:
        return f"Parameter(shape={self._shape})"

    def tree_flatten(self):
        raw_params = (self._raw_value,)
        aux = (self.bijector, self.prior, self._trainable)
        return raw_params, aux

    @classmethod
    def tree_unflatten(cls, aux_data, raw_params):
        bijector, prior, trainable = aux_data
        raw_value = raw_params[0]
        value = bijector(raw_value)
        return cls(value, bijector, prior, trainable)


# @jtu.register_pytree_node_class
class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self._training = True

    def train(self):
        self._training = True
        for module in self._modules.values():
            module.train()
        return self

    def eval(self):
        self._training = False
        for module in self._modules.values():
            module.eval()
        return self

    def trainable(self, is_trainable: bool = True):
        for param in self._parameters.values():
            param.trainable(is_trainable)
        for module in self._modules.values():
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
        else:
            raise AttributeError(f"Module has no attribute {key}")

    def get_raw_parameters(self, raw_dict=True):
        params = jtu.tree_map(lambda x: x, self._parameters)  # copy
        for module_name, module in self._modules.items():
            params[module_name] = module.get_raw_parameters(raw_dict=False)

        if raw_dict:
            return jtu.tree_map(lambda x: x.get_raw_value(), params, is_leaf=is_parameter)
        else:
            return params

    def get_parameters(self, raw_dict=True):
        params = jtu.tree_map(lambda x: x, self._parameters)  # copy
        for module_name, module in self._modules.items():
            params[module_name] = module.get_parameters(raw_dict=False)

        if raw_dict:
            return jtu.tree_map(lambda x: x.get_value(), params, is_leaf=is_parameter)
        else:
            return params

    def set_raw_parameters(self, raw_params):
        raw_params = jtu.tree_map(lambda x: x, raw_params)  # copy
        for module_name, module in self._modules.items():
            module.set_raw_parameters(raw_params[module_name])
            raw_params.pop(module_name)

        for param_name, param in self._parameters.items():
            param.set_raw_value(raw_params[param_name])

    def set_parameters(self, params):
        params = jtu.tree_map(lambda x: x, params)  # copy
        for module_name, module in self._modules.items():
            module.set_parameters(params[module_name])
            params.pop(module_name)

        for param_name, param in self._parameters.items():
            param.set_value(params[param_name])

    def initialize(self, key):
        key1, key2 = jax.random.split(key)

        for module in self._modules.values():
            module.initialize(key1)
            key1 = jax.random.split(key1, num=1)[0]

        for param in self._parameters.values():
            param.initialize(key2)
            key2 = jax.random.split(key2, num=1)[0]
        return self

    def log_prob(self, reduce=True):
        log_probs = {}
        for module_name, module in self._modules.items():
            log_probs[module_name] = module.log_prob(reduce=False)

        for param_name, param in self._parameters.items():
            log_probs[param_name] = param.log_prob()

        if reduce:
            return ravel_pytree(log_probs)[0].sum()
        else:
            return log_probs

    # TODO: fix this to work with pytree
    # def tree_flatten(self):

    # @classmethod
    # def tree_unflatten(cls, aux_data, flat_values):
    #     self = cls()
    #     self._training = aux_data["___training"]
    #     all_values = aux_data["___unravel_fn"](flat_values)
    #     for param_name, aux in aux_data["___parameters"].items():
    #         param = Parameter.tree_unflatten(aux, all_values["___parameters"][param_name])
    #         self._parameters[param_name] = param

    #     for module_name, aux in aux_data["___modules"].items():
    #         module = Module.tree_unflatten(aux, (all_values["___modules"][module_name],))
    #         module.training = aux["___training"]
    #         self._modules[module_name] = module

    #     return self
