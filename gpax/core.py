import os

import jax
import jax.tree_util as jtu
import jax.numpy as jnp

import gpax.bijectors as gb
import gpax.distributions as gd

import inspect


class Base:
    """
    This class provides a skeleton for all classes.
    """

    def initialize_params(self, key=None, aux=None):
        init_params = self.__initialize_params__(aux)
        init_params = jtu.tree_map(lambda x: jnp.asarray(x), init_params)
        if key is None:
            params = init_params
        else:
            params = sample_params(init_params, self.priors, self.constraints, key)

        params = self.__post_initialize_params__(params, aux)
        return params

    def __post_initialize_params__(self, params, aux):
        return params

    def __initialize_params__(self, aux):
        raise NotImplementedError("This method must be implemented by a subclass.")


def sample_params(params, priors, bijectors, key, generic_sampler=jax.random.normal):
    seeds = seeds_like(params, key)

    def _randomize(param, prior, bijector, seed):
        if prior is None:
            sample = generic_sampler(seed, param.shape)
            return bijector(sample)
        else:
            return prior.sample(seed=seed)

    return jtu.tree_map(
        lambda prior, param, bijector, seed: _randomize(param, prior, bijector, seed),
        priors,
        params,
        bijectors,
        seeds,
        is_leaf=lambda x: x is None,
    )


def get_raw_log_prior(priors, params, bijectors):
    def _get_raw_log_prior(prior, param, bijector):
        if prior is None:
            return jnp.zeros_like(param)
        else:
            return prior.log_prob(param) - bijector.inverse_log_jacobian(param)

    return jtu.tree_map(
        lambda prior, param, bijector: _get_raw_log_prior(prior, param, bijector), priors, params, bijectors
    )


def seeds_like(params, key):
    values, treedef = jtu.tree_flatten(params)
    keys = [key for key in jax.random.split(key, len(values))]
    return jtu.tree_unflatten(treedef, keys)


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
