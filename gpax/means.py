from gpax.core import Parameter
from gpax.core import Module
from chex import dataclass
from typing import Union


class Mean(Module):
    """
    A meta class to define a mean function.
    """

    pass


@dataclass
class Scalar(Mean):
    value: Union[Parameter, float] = 0.0

    def __post_init__(self):
        if not isinstance(self.value, Parameter):
            self.value = Parameter(self.value)

    def __call__(self, y=None):
        return self.value()

    def __get_params__(self):
        return {"value": self.value}

    def set_params(self, raw_params):
        self.value.set(raw_params["value"])


@dataclass
class Average(Mean):
    def __call__(self, y):
        return y.mean()

    def __get_params__(self):
        return {}

    def set_params(self, raw_params):
        pass
