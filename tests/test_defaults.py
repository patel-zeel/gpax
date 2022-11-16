from gpax import DEFAULT_JITTER

from gpax.defaults import set_default_jitter, get_default_jitter
import gpax.distributions as gd
import gpax.bijectors as gb


def test_default_prior():
    assert isinstance(gd.get_default_prior(), gd.Normal)
    gd.set_default_prior(gd.Uniform)
    assert isinstance(gd.get_default_prior(), gd.Uniform)


def test_default_bijector():
    assert isinstance(gb.get_default_bijector(), gb.Identity)
    gb.set_default_bijector(gb.Softplus)
    assert isinstance(gb.get_default_bijector(), gb.Softplus)


def test_positive_bijector():
    assert isinstance(gb.get_positive_bijector(), gb.Exp)
    gb.set_positive_bijector(gb.Softplus)
    assert isinstance(gb.get_positive_bijector(), gb.Softplus)


def test_default_jitter():
    assert get_default_jitter() == DEFAULT_JITTER
    set_default_jitter(1e-4)
    assert get_default_jitter() == 1e-4
