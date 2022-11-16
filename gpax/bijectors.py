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
    from gpax.core import Parameter, Mean

NEAR_ZERO = 1e-10


def invert_bijector(bijector):
    bijector = deepcopy(bijector)
    bijector.forward_fn, bijector.inverse_fn = bijector.inverse_fn, bijector.forward_fn
    return bijector


class Bijector:
    def __call__(self, ele):
        if isinstance(ele, Distribution):
            return TransformedDistribution(distribution=ele, bijector=self)
        else:
            return self.forward_fn(ele)

    def forward(self, ele):
        return self(ele)

    def inverse(self, ele):
        if isinstance(ele, Distribution):
            return TransformedDistribution(distribution=ele, bijector=invert_bijector(self))
        else:
            return self.inverse_fn(ele)

    def log_jacobian(self, value):
        def _log_jacobian(value):
            return jnp.log(jax.jacobian(self.forward)(value))

        return vectorized_fn(_log_jacobian, value, value.shape)

    def inverse_log_jacobian(self, array):
        def _inverse_log_jacobian(value):
            return jnp.log(jax.jacobian(self.inverse)(value))

        return vectorized_fn(_inverse_log_jacobian, array, array.shape)


@dataclass
class Log(Bijector):
    forward_fn: callable = jnp.log
    inverse_fn: callable = jnp.exp
    in_upper: float = jnp.inf
    in_lower: float = NEAR_ZERO
    out_upper: float = jnp.inf
    out_lower: float = -jnp.inf


@dataclass
class Exp(Bijector):
    forward_fn: callable = jnp.exp
    inverse_fn: callable = jnp.log
    in_upper: float = jnp.inf
    in_lower: float = -jnp.inf
    out_upper: float = jnp.inf
    out_lower: float = 0.0


@dataclass
class Sigmoid(Bijector):
    forward_fn: callable = jax.nn.sigmoid
    inverse_fn: callable = jsp.special.logit
    in_upper: float = jnp.inf
    in_lower: float = -jnp.inf
    out_upper: float = 1.0
    out_lower: float = 0.0


@dataclass
class Identity(Bijector):
    forward_fn: callable = lambda x: x
    inverse_fn: callable = lambda x: x
    in_upper: float = jnp.inf
    in_lower: float = -jnp.inf
    out_upper: float = jnp.inf
    out_lower: float = -jnp.inf


@dataclass
class SquarePlus(Bijector):
    forward_fn: callable = lambda x: 0.5 * (x + jnp.sqrt(jnp.square(x) + 4.0))
    inverse_fn: callable = lambda x: x - 1 / x
    in_upper: float = jnp.inf
    in_lower: float = -jnp.inf
    out_upper: float = jnp.inf
    out_lower: float = 0.0


@dataclass
class Softplus(Bijector):
    forward_fn: callable = jax.nn.softplus
    inverse_fn: callable = lambda x: jnp.log(jnp.exp(x) - 1)
    in_upper: float = jnp.inf
    in_lower: float = -jnp.inf
    out_upper: float = jnp.inf
    out_lower: float = 0.0


@dataclass
class White(Bijector):
    kernel_fn: callable = None
    X_inducing: Parameter = None
    mean: Mean = None

    def forward_fn(self, white_value):
        positive_bijector = get_positive_bijector()
        X_inducing = self.X_inducing()
        mean = self.mean()
        covariance = self.kernel_fn(X_inducing, X_inducing)
        stable_covariance = add_to_diagonal(covariance, 0.0, get_default_jitter())
        cholesky = jnp.linalg.cholesky(stable_covariance)
        raw_value = cholesky @ white_value + mean
        return positive_bijector(raw_value)

    def inverse_fn(self, value):
        positive_bijector = get_positive_bijector()
        X_inducing = self.X_inducing()
        mean = self.mean()

        value = repeat_to_size(value, X_inducing.shape[0])
        raw_value = positive_bijector.inverse(value)
        covariance = self.kernel_fn(X_inducing, X_inducing)
        stable_covariance = add_to_diagonal(covariance, 0.0, get_default_jitter())
        cholesky = jnp.linalg.cholesky(stable_covariance)

        raw_value_bar = raw_value - mean
        white_value = jsp.linalg.solve_triangular(cholesky, raw_value_bar, lower=True)
        return white_value


all_bijectors = {
    "Log": Log,
    "Exp": Exp,
    "Sigmoid": Sigmoid,
    "Identity": Identity,
    "SquarePlus": SquarePlus,
    "Softplus": Softplus,
}


def set_default_bijector(bijector):
    assert inspect.isclass(bijector)
    os.environ["DEFAULT_BIJECTOR"] = bijector.__name__


def get_default_bijector():
    bijector = all_bijectors[os.environ["DEFAULT_BIJECTOR"]]
    return bijector()


def set_positive_bijector(bijector):
    assert inspect.isclass(bijector)
    os.environ["POSITIVE_BIJECTOR"] = bijector.__name__


def get_positive_bijector():
    bijector = all_bijectors[os.environ["POSITIVE_BIJECTOR"]]
    return bijector()
