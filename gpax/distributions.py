from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.scipy as jsp


class Distribution:
    @abstractmethod
    def sample(self, seed, sample_shape=()):
        NotImplementedError("This method must be implemented by a subclass.")

    @abstractmethod
    def log_prob(self, value):
        NotImplementedError("This method must be implemented by a subclass.")


class TransformedDistribution(Distribution):
    def __init__(self, distribution, bijector):
        self.distribution = distribution
        self.bijector = bijector

    def sample(self, seed, sample_shape=()):
        return self.bijector(self.distribution.sample(seed, sample_shape))

    def log_prob(self, value):
        return self.distribution.log_prob(self.bijector.inverse(value)) + self.bijector.inverse_log_jacobian(value)


class Zero(Distribution):
    def sample(self, seed, sample_shape=()):
        NotImplementedError("Zero distribution cannot be sampled.")

    def log_prob(self, value):
        return jnp.zeros_like(value)


class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed, sample_shape=()):
        return self.loc + jax.random.normal(seed, sample_shape) * self.scale

    def log_prob(self, value):
        return jsp.stats.norm.logpdf(value, loc=self.loc, scale=self.scale)


class Beta(Distribution):
    def __init__(self, concentration0, concentration1):
        self.concentration0 = concentration0
        self.concentration1 = concentration1

    def sample(self, seed, sample_shape=()):
        return jax.random.beta(seed, a=self.concentration1, b=self.concentration0, shape=sample_shape)

    def log_prob(self, value):
        return jsp.stats.beta.logpdf(value, a=self.concentration1, b=self.concentration0)
