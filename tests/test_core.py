import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
from gpax.core import (
    Module,
    Parameter,
    get_default_prior,
    set_default_prior,
    get_default_jitter,
    set_default_jitter,
    set_positive_bijector,
    get_positive_bijector,
)
from tests.utils import assert_same_pytree

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

import pytest

# jax 64 bit mode
jax.config.update("jax_enable_x64", True)


def test_default_jitter():
    jitter = get_default_jitter()
    assert jitter == 1e-6
    set_default_jitter(1e-3)
    assert get_default_jitter() == 1e-3


def test_positive_bijector():
    cached_positive_bijector = get_positive_bijector()

    set_positive_bijector(tfb.Exp())
    assert isinstance(get_positive_bijector(), tfb.Exp)

    set_positive_bijector(cached_positive_bijector)


def test_default_prior():
    default_prior = get_default_prior()
    assert isinstance(default_prior, tfd.Normal)
    assert default_prior.loc == 0.0
    assert default_prior.scale == 1.0

    n = 1000

    p = Parameter(jnp.ones((n,)))
    p.initialize(jax.random.PRNGKey(0))
    assert (p() < 0.0).sum() > 0

    set_default_prior(tfd.Gamma(concentration=1.0, rate=1.0))
    p = Parameter(jnp.ones((n,)))
    p.initialize(jax.random.PRNGKey(0))
    assert (p() < 0.0).sum() == 0


@pytest.mark.parametrize("value", [1.0, jnp.array(1.0), jnp.array([1.0, 0.2])])
@pytest.mark.parametrize("bijector", [None, tfb.Exp(), tfb.Softplus()])
@pytest.mark.parametrize("prior", [None, tfd.Normal(loc=0.0, scale=1.0), tfd.Gamma(concentration=1.0, rate=1.0)])
def test_parameter(value, bijector, prior):
    p = Parameter(value, bijector=bijector, prior=prior)
    assert jnp.allclose(p(), jnp.asarray(value))

    if bijector is None:
        if prior is None:
            bijector = tfb.Identity()
        else:
            bijector = prior._default_event_space_bijector()

    assert jnp.allclose(p.get_raw_value(), bijector.inverse(jnp.asarray(value)))

    if prior is None:
        assert p.log_prob().sum() == 0.0
    else:
        assert p.log_prob().sum() == prior.log_prob(value).sum()


def test_module():
    m1 = Module()
    m1.p1 = Parameter(1.0)
    m1.m2 = Module()
    m1.m3 = Module()
    m1.m2.p2 = Parameter(2.0)
    m1.m3.p3 = Parameter(3.0)

    params = m1.get_parameters(raw_dict=True)
    assert_same_pytree(params, {"p1": 1.0, "m2": {"p2": 2.0}, "m3": {"p3": 3.0}})

    set_params = {"p1": 5.0, "m2": {"p2": 4.0}, "m3": {"p3": 9.0}}
    m1.set_parameters(set_params)
    assert_same_pytree(m1.get_parameters(raw_dict=True), set_params)

    assert m1.p1() == 5.0
    assert m1.m2.p2() == 4.0
    assert m1.m3.p3() == 9.0
