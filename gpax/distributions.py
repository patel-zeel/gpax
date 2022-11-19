from __future__ import annotations
import os
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass

import inspect

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bijectors import Bijector


class Distribution:
    @abstractmethod
    def sample(self, seed, sample_shape):
        NotImplementedError("This method must be implemented by a subclass.")

    @abstractmethod
    def log_prob(self, value):
        NotImplementedError("This method must be implemented by a subclass.")


@dataclass
class TransformedDistribution(Distribution):
    distribution: Distribution
    bijector: Bijector

    def sample(self, seed, sample_shape):
        return self.bijector(self.distribution.sample(seed, sample_shape))

    def log_prob(self, value):
        log_prob = self.distribution.log_prob(self.bijector.inverse(value))
        return log_prob + self.bijector.inverse_log_jacobian(value)


@dataclass
class Normal(Distribution):
    loc: float = 0.0
    scale: float = 1.0

    def sample(self, seed, sample_shape):
        return self.loc + jax.random.normal(seed, sample_shape) * self.scale

    def log_prob(self, value):
        return jsp.stats.norm.logpdf(value, loc=self.loc, scale=self.scale)


@dataclass
class Uniform(Distribution):
    low: float = 0.0
    high: float = 1.0

    def sample(self, seed, sample_shape):
        return jax.random.uniform(seed, sample_shape, minval=self.low, maxval=self.high)

    def log_prob(self, value):
        return jsp.stats.uniform.logpdf(value, loc=self.low, scale=self.high - self.low)


@dataclass
class Gamma(Distribution):
    concentration: float
    rate: float

    def sample(self, seed, sample_shape):
        return jax.random.gamma(seed, shape=sample_shape, a=self.concentration) / self.rate

    def log_prob(self, value):
        return jsp.stats.gamma.logpdf(value, a=self.concentration, scale=1 / self.rate)


@dataclass
class Beta(Distribution):
    concentration0: float
    concentration1: float

    def sample(self, seed, sample_shape):
        return jax.random.beta(seed, a=self.concentration1, b=self.concentration0, shape=sample_shape)

    def log_prob(self, value):
        return jsp.stats.beta.logpdf(value, a=self.concentration1, b=self.concentration0)


@dataclass
class Exponential(Distribution):
    rate: float

    def sample(self, seed, sample_shape):
        return jax.random.exponential(seed, shape=sample_shape) / self.rate

    def log_prob(self, value):
        return jsp.stats.expon.logpdf(value, scale=1 / self.rate)


@dataclass
class Frechet(Distribution):
    rate: float
    dim: int

    def sample(self, seed, sample_shape):
        samples = jax.random.uniform(key=seed, shape=sample_shape)
        return self.inverse_cdf(samples)

    def inverse_cdf(self, x):
        return (-jnp.log(x) / self.rate) ** (-2 / self.dim)

    def cdf(self, x):
        return jnp.exp(-self.rate * x ** (-self.dim / 2))

    def log_prob(self, value):
        prefix = jnp.log(self.dim / 2 * self.rate * value ** (-self.dim / 2 - 1))
        return prefix + (-self.rate * value ** (-self.dim / 2))


all_distributions = {
    "Normal": Normal,
    "Uniform": Uniform,
    "Gamma": Gamma,
    "Beta": Beta,
    "Exponential": Exponential,
    "Frechet": Frechet,
}


# Getter and Setters
def set_default_prior(prior):
    assert inspect.isclass(prior)
    os.environ["DEFAULT_PRIOR"] = prior.__name__


def get_default_prior():
    prior = all_distributions[os.environ["DEFAULT_PRIOR"]]
    return prior()
