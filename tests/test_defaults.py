from gpax import DEFAULT_JITTER
from gpax.core import (
    set_default_prior,
    get_default_prior,
    set_default_bijector,
    get_default_bijector,
    set_positive_bijector,
    get_positive_bijector,
    set_default_jitter,
    get_default_jitter,
)
import gpax.distributions as gd
import gpax.bijectors as gb


def test_default_prior():
    assert isinstance(get_default_prior(), gd.Normal)
    set_default_prior(gd.Uniform)
    assert isinstance(get_default_prior(), gd.Uniform)


def test_default_bijector():
    assert isinstance(get_default_bijector(), gb.Identity)
    set_default_bijector(gb.Softplus)
    assert isinstance(get_default_bijector(), gb.Softplus)


def test_positive_bijector():
    assert isinstance(get_positive_bijector(), gb.Exp)
    set_positive_bijector(gb.Softplus)
    assert isinstance(get_positive_bijector(), gb.Softplus)


def test_default_jitter():
    assert get_default_jitter() == DEFAULT_JITTER
    set_default_jitter(1e-4)
    assert get_default_jitter() == 1e-4
