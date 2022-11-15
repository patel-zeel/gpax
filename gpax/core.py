import os

import jax
import jax.tree_util as jtu
import jax.numpy as jnp


from gpax.utils import constrain, unconstrain
import gpax.bijectors as gb
import gpax.distributions as gd
from chex import dataclass
from dataclasses import field

import inspect

pytree = jtu.register_pytree_node_class


# Core functions


def set_default_prior(prior):
    assert inspect.isclass(prior)
    os.environ["DEFAULT_PRIOR"] = prior.__name__


def get_default_prior():
    prior = getattr(gd, os.environ["DEFAULT_PRIOR"])
    return prior()


def set_default_bijector(bijector):
    assert inspect.isclass(bijector)
    os.environ["DEFAULT_BIJECTOR"] = bijector.__name__


def get_default_bijector():
    bijector = getattr(gb, os.environ["DEFAULT_BIJECTOR"])
    return bijector()


def set_positive_bijector(bijector):
    assert inspect.isclass(bijector)
    os.environ["POSITIVE_BIJECTOR"] = bijector.__name__


def get_positive_bijector():
    bijector = getattr(gb, os.environ["POSITIVE_BIJECTOR"])
    return bijector()


def set_default_jitter(jitter):
    os.environ["DEFAULT_JITTER"] = str(jitter)


def get_default_jitter():
    return float(os.environ["DEFAULT_JITTER"])


set_default_prior(gd.Normal)
set_default_bijector(gb.Identity)
set_positive_bijector(gb.Exp)

# Core classes


@dataclass
class Module:
    """
    This class provides a skeleton for all classes.
    """

    default_sampler: gd.Distribution = get_positive_bijector()(gd.Normal(0.0, 1.0))
    modules: dict = field(default_factory=lambda: {})

    def init_params(self, key):
        def _randomize(prior, flat_param, key):
            if prior is None:
                return self.default_sampler.sample(key, flat_param.shape)
            elif isinstance(prior, gd.Fixed):
                return flat_param
            else:
                return prior.sample(key, flat_param.shape)

        params = self.get_params()
        flat_params, treedef = jtu.tree_flatten(params)
        priors, _ = jtu.tree_flatten(self.priors, is_leaf=lambda x: isinstance(x, gd.Distribution))
        keys = [key for key in jax.random.split(key, len(flat_params))]

        random_values = jtu.tree_map(
            lambda prior, flat_param, key: _randomize(prior, flat_param, key), priors, flat_params, keys
        )
        params = treedef.unflatten(random_values)
        self.set_params(params)
        self._post_init_params()

    def _post_init_params(self):
        pass

    def constrain(self):
        params = self.get_params()
        params = constrain(params, self.constraints)
        self.set_params(params)

    def unconstrain(self):
        params = self.get_params()
        params = unconstrain(params, self.constraints)
        self.set_params(params)


@dataclass
class Model(Module):
    """
    This class provides a skeleton for all models.
    """

    pass


@dataclass
class Kernel(Module):
    """
    A meta class to define a kernel.
    """

    pass


@dataclass
class Likelihood(Module):
    """
    A meta class to define a likelihood.
    """

    pass


@dataclass
class Mean(Module):
    """
    A meta class to define a mean function.
    """

    pass
