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
        if self.distribution.__class__.__name__ == "Zero":
            return jnp.zeros_like(value)
        else:
            return self.distribution.log_prob(self.bijector.inverse(value)) + self.bijector.inverse_log_jacobian(value)


class NoPrior(Distribution):
    def sample(self, seed, sample_shape=()):
        return jax.random.normal(seed, sample_shape)

    def log_prob(self, value):
        return jnp.zeros_like(value)


class Normal(Distribution):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def sample(self, seed, sample_shape=()):
        return self.loc + jax.random.normal(seed, sample_shape) * self.scale

    def log_prob(self, value):
        return jsp.stats.norm.logpdf(value, loc=self.loc, scale=self.scale)


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate

    def sample(self, seed, sample_shape=()):
        return jax.random.gamma(seed, shape=sample_shape, a=self.concentration) / self.rate

    def log_prob(self, value):
        return jsp.stats.gamma.logpdf(value, a=self.concentration, scale=1 / self.rate)


class Beta(Distribution):
    def __init__(self, concentration0, concentration1):
        self.concentration0 = concentration0
        self.concentration1 = concentration1

    def sample(self, seed, sample_shape=()):
        return jax.random.beta(seed, a=self.concentration1, b=self.concentration0, shape=sample_shape)

    def log_prob(self, value):
        return jsp.stats.beta.logpdf(value, a=self.concentration1, b=self.concentration0)


class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def sample(self, seed, sample_shape=()):
        return jax.random.exponential(seed, shape=sample_shape) / self.rate

    def log_prob(self, value):
        return jsp.stats.expon.logpdf(value, scale=1 / self.rate)


class Frechet(Distribution):
    def __init__(self, rate, dim):
        self.rate = rate
        self.dim = dim

    def sample(self, seed, sample_shape=()):
        samples = jax.random.uniform(key=seed, shape=sample_shape)
        return self.inverse_cdf(samples)

    def inverse_cdf(self, x):
        return (-jnp.log(x) / self.rate) ** (-2 / self.dim)

    def cdf(self, x):
        return jnp.exp(-self.rate * x ** (-self.dim / 2))

    def log_prob(self, value):
        prefix = jnp.log(self.dim / 2 * self.rate * value ** (-self.dim / 2 - 1))
        return prefix + (-self.rate * value ** (-self.dim / 2))
