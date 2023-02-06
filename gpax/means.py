import jax.numpy as jnp

from gpax.core import Parameter
from gpax.core import Module

from jaxtyping import Array, Float


class Mean(Module):
    """
    A meta class to define a mean function.
    """

    pass


class Scalar(Mean):
    def __init__(self, value: Float[Array, "1"] = 0.0):
        super(Scalar, self).__init__()
        self.value = Parameter(jnp.asarray(value))

    def __call__(self, y):
        return self.value.get_value()  # or self.value()

    def __repr__(self) -> str:
        return f"Scalar"


class Average(Mean):
    def __call__(self, y):
        return y.mean()

    def __repr__(self) -> str:
        return f"Average"
