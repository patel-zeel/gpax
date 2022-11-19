from __future__ import annotations
from typing import TYPE_CHECKING

from copy import deepcopy

import os
import jax
import jax.numpy as jnp
import jax.scipy as jsp


from gpax.defaults import get_default_jitter
from gpax.distributions import TransformedDistribution, Distribution
from gpax.utils import vectorized_fn, add_to_diagonal, repeat_to_size
from chex import dataclass

import inspect

if TYPE_CHECKING:
    from gpax.core import Parameter
    from gpax.models import Model


def invert_bijector(bijector):
    return bijector_pairs[bijector.__class__]()


class Bijector:
    def __call__(self, ele):
        if ele is None:
            return None
        elif isinstance(ele, TransformedDistribution):
            if bijector_pairs[type(ele.bijector)] is type(self):
                return ele.distribution
            else:
                return TransformedDistribution(distribution=ele, bijector=self)
        elif isinstance(ele, Distribution):
            return TransformedDistribution(distribution=ele, bijector=self)
        else:
            return self._forward_fn(ele)

    def forward(self, ele):
        return self(ele)

    def inverse(self, ele):
        if ele is None:
            return None
        elif isinstance(ele, TransformedDistribution):
            if type(ele.bijector) is type(self):
                return ele.distribution
        elif isinstance(ele, Distribution):
            return TransformedDistribution(distribution=ele, bijector=invert_bijector(self))
        else:
            return self._inverse_fn(ele)

    def log_jacobian(self, value):
        def _log_jacobian(value):
            return jnp.log(jax.jacobian(self.forward)(value))

        return vectorized_fn(_log_jacobian, value, value.shape)

    def inverse_log_jacobian(self, array):
        def _inverse_log_jacobian(value):
            return jnp.log(jax.jacobian(self.inverse)(value))

        return vectorized_fn(_inverse_log_jacobian, array, array.shape)

    def in_limits(self):
        return (self._in_lower, self._in_upper)

    def out_limits(self):
        return (self._out_lower, self._out_upper)


@dataclass
class Exp(Bijector):
    _forward_fn: callable = jnp.exp
    _inverse_fn: callable = jnp.log
    _in_upper: float = jnp.inf
    _in_lower: float = -jnp.inf
    _out_upper: float = jnp.inf
    _out_lower: float = 0.0


@dataclass
class Log(Bijector):
    _forward_fn: callable = jnp.log
    _inverse_fn: callable = jnp.exp
    _in_upper: float = jnp.inf
    _in_lower: float = 0.0
    _out_upper: float = jnp.inf
    _out_lower: float = -jnp.inf


@dataclass
class Sigmoid(Bijector):
    _forward_fn: callable = jax.nn.sigmoid
    _inverse_fn: callable = jsp.special.logit
    _in_upper: float = jnp.inf
    _in_lower: float = -jnp.inf
    _out_upper: float = 1.0
    _out_lower: float = 0.0


@dataclass
class Logit(Bijector):
    _forward_fn: callable = jsp.special.logit
    _inverse_fn: callable = jax.nn.sigmoid
    _in_upper: float = 1.0
    _in_lower: float = 0.0
    _out_upper: float = jnp.inf
    _out_lower: float = -jnp.inf


@dataclass
class Identity(Bijector):
    _forward_fn: callable = lambda x: x
    _inverse_fn: callable = lambda x: x
    _in_upper: float = jnp.inf
    _in_lower: float = -jnp.inf
    _out_upper: float = jnp.inf
    _out_lower: float = -jnp.inf


@dataclass
class Square(Bijector):
    _forward_fn: callable = jnp.square
    _inverse_fn: callable = jnp.sqrt
    _in_upper: float = jnp.inf
    _in_lower: float = -jnp.inf
    _out_upper: float = jnp.inf
    _out_lower: float = 0.0


@dataclass
class Sqrt(Bijector):
    _forward_fn: callable = jnp.sqrt
    _inverse_fn: callable = jnp.square
    _in_upper: float = jnp.inf
    _in_lower: float = 0.0
    _out_upper: float = jnp.inf
    _out_lower: float = 0.0


@dataclass
class SquarePlus(Bijector):
    _forward_fn: callable = lambda x: 0.5 * (x + jnp.sqrt(jnp.square(x) + 4.0))
    _inverse_fn: callable = lambda x: x - 1 / x
    _in_upper: float = jnp.inf
    _in_lower: float = -jnp.inf
    _out_upper: float = jnp.inf
    _out_lower: float = 0.0


@dataclass
class InverseSquarePlus(Bijector):
    _forward_fn: callable = lambda x: x - 1 / x
    _inverse_fn: callable = lambda x: 0.5 * (x + jnp.sqrt(jnp.square(x) + 4.0))
    _in_upper: float = jnp.inf
    _in_lower: float = 0.0
    _out_upper: float = jnp.inf
    _out_lower: float = -jnp.inf


@dataclass
class Softplus(Bijector):
    _forward_fn: callable = jax.nn.softplus
    _inverse_fn: callable = lambda x: jnp.log(jnp.exp(x) - 1)
    _in_upper: float = jnp.inf
    _in_lower: float = -jnp.inf
    _out_upper: float = jnp.inf
    _out_lower: float = 0.0


@dataclass
class InverseSoftplus(Bijector):
    _forward_fn: callable = lambda x: jnp.log(jnp.exp(x) - 1)
    _inverse_fn: callable = jax.nn.softplus
    _in_upper: float = jnp.inf
    _in_lower: float = 0.0
    _out_upper: float = jnp.inf
    _out_lower: float = -jnp.inf


def white_forward_fn(self, value):
    positive_bijector = get_positive_bijector()
    X_inducing = self.X_inducing()
    mean = self.latent_gp.mean()

    value = repeat_to_size(value, X_inducing.shape[0])
    raw_value = positive_bijector.inverse(value)
    covariance = self.latent_gp.kernel(X_inducing, X_inducing)
    stable_covariance = add_to_diagonal(covariance, 0.0, get_default_jitter())
    cholesky = jnp.linalg.cholesky(stable_covariance)

    raw_value_bar = raw_value - mean
    white_value = jsp.linalg.solve_triangular(cholesky, raw_value_bar, lower=True)
    return white_value


def white_inverse_fn(self, white_value):
    positive_bijector = get_positive_bijector()
    X_inducing = self.X_inducing()
    mean = self.latent_gp.mean()
    covariance = self.latent_gp.kernel(X_inducing, X_inducing)
    stable_covariance = add_to_diagonal(covariance, 0.0, get_default_jitter())
    cholesky = jnp.linalg.cholesky(stable_covariance)
    raw_value = cholesky @ white_value + mean
    return positive_bijector(raw_value)


@dataclass
class White(Bijector):
    latent_gp: Model = None
    X_inducing: Parameter = None
    _in_upper: float = jnp.inf
    _in_lower: float = 0.0
    _out_upper: float = jnp.inf
    _out_lower: float = -jnp.inf

    def __post_init__(self):
        self._forward_fn = lambda value: white_forward_fn(self, value)
        self._inverse_fn = lambda white_value: white_inverse_fn(self, white_value)


@dataclass
class InverseWhite(Bijector):
    latent_gp: Model = None
    X_inducing: Parameter = None
    _in_upper: float = jnp.inf
    _in_lower: float = -jnp.inf
    _out_upper: float = jnp.inf
    _out_lower: float = 0.0

    def __post_init__(self):
        self._forward_fn = lambda white_value: white_inverse_fn(self, white_value)
        self._inverse_fn = lambda value: white_forward_fn(self, value)


bijector_pairs = {
    Exp: Log,
    Sigmoid: Logit,
    Identity: Identity,
    Square: Sqrt,
    SquarePlus: InverseSquarePlus,
    Softplus: InverseSoftplus,
    White: InverseWhite,
}
bijector_pairs.update({v: k for k, v in bijector_pairs.items()})


def set_default_bijector(bijector):
    assert inspect.isclass(bijector)
    os.environ["DEFAULT_BIJECTOR"] = bijector.__name__


def get_default_bijector():
    bijector = globals()[os.environ["DEFAULT_BIJECTOR"]]
    return bijector()


def set_positive_bijector(bijector):
    assert inspect.isclass(bijector)
    os.environ["POSITIVE_BIJECTOR"] = bijector.__name__


def get_positive_bijector():
    bijector = globals()[os.environ["POSITIVE_BIJECTOR"]]
    return bijector()
