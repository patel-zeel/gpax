from gpax.core import Parameter
import gpax.distributions as gd
from gpax.core import Module
from chex import dataclass


class Mean(Module):
    """
    A meta class to define a mean function.
    """

    pass


@dataclass
class Scalar(Mean):
    value: float = 1.0
    value_prior: gd.Distribution = None

    def __post_init__(self):
        self.value = Parameter(self.value, prior=self.value_prior)

    def __call__(self, y=None):
        return self.value()

    def __get_params__(self):
        return {"value": self.value}

    def set_params(self, params):
        self.value.set(params["value"])


@dataclass
class Average(Mean):
    def __call__(self, y):
        return y.mean()

    def __get_params__(self):
        return {}

    def set_params(self, params):
        pass
