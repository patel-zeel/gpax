import jax
import jax.tree_util as jtu
import jax.numpy as jnp

from gpax.core import Parameter, Module
import gpax.distributions as gd
import gpax.bijectors as gb
from gpax.utils import squared_distance, distance, repeat_to_size
from chex import dataclass


class Kernel(Module):
    def __add__(self, other):
        return Sum(k1=self, k2=other)

    def __mul__(self, other):
        return Product(k1=self, k2=other)


@dataclass
class MathKernel(Kernel):
    k1: Kernel = None
    k2: Kernel = None

    def __get_params__(self):
        return {"k1": self.k1.__get_params__(), "k2": self.k2.__get_params__()}

    def set_params(self, params):
        self.k1.set_params(params["k1"])
        self.k2.set_params(params["k2"])


class Sum(MathKernel):
    def __call__(self, X1, X2):
        return self.k1(X1, X2) + self.k2(X1, X2)


class Product(MathKernel):
    def __call__(self, X1, X2):
        return self.k1(X1, X2) * self.k2(X1, X2)


@dataclass
class SubKernel(Kernel):
    input_dim: int = None
    active_dims: list = None
    ARD: bool = True

    def __post_init__(self):
        if self.active_dims is None:
            assert self.input_dim is not None, "input_dim must be specified if active_dims is None"
            self.active_dims = list(range(self.input_dim))
        if self.input_dim is None:
            self.input_dim = len(self.active_dims)
        assert len(self.active_dims) == self.input_dim, "active_dims must have the same length as input_dim."

    def __call__(self, X1, X2):
        X1_slice = X1[:, self.active_dims]
        X2_slice = X2[:, self.active_dims]
        return self.call(X1_slice, X2_slice)


@dataclass
class Smooth(SubKernel):
    lengthscale: float = 1.0
    lengthscale_prior: gd.Distribution = None
    scale: float = 1.0
    scale_prior: gd.Distribution = None

    def __post_init__(self):
        super(Smooth, self).__post_init__()
        if self.ARD:
            lengthscale = jnp.asarray(self.lengthscale).squeeze()
            assert lengthscale.size in (
                1,
                self.input_dim,
            ), "lengthscale must be a scalar or an array of shape (input_dim,)."
            lengthscale = repeat_to_size(lengthscale, self.input_dim)
        else:
            lengthscale = jnp.asarray(self.lengthscale).squeeze()
            assert lengthscale.shape == (), "lengthscale must be a scalar when ARD=False."

        self.lengthscale = Parameter(lengthscale, bijector=gb.get_positive_bijector(), prior=self.lengthscale_prior)
        self.scale = Parameter(self.scale, bijector=gb.get_positive_bijector(), prior=self.scale_prior)

    def call(self, X1, X2):
        kernel_fn = self.call_on_a_pair
        kernel_fn = jax.vmap(kernel_fn, in_axes=(None, 0))
        kernel_fn = jax.vmap(kernel_fn, in_axes=(0, None))
        return kernel_fn(X1, X2)

    def __get_params__(self):
        return {"lengthscale": self.lengthscale, "scale": self.scale}

    def set_params(self, params):
        self.lengthscale.set(params["lengthscale"])
        self.scale.set(params["scale"])


class RBF(Smooth):
    def call_on_a_pair(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        sqr_dist = squared_distance(x1_scaled, x2_scaled)
        return ((self.scale() ** 2) * jnp.exp(-0.5 * sqr_dist)).squeeze()

    def __repr__(self) -> str:
        return "RBF"


ExpSquared = RBF
SquaredExp = RBF


class Matern12(Smooth):
    def call_on_a_pair(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        dist = distance(x1_scaled, x2_scaled)
        return ((self.scale() ** 2) * jnp.exp(-dist)).squeeze()

    def __repr__(self) -> str:
        return "Matern12"


class Matern32(Smooth):
    def call_on_a_pair(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        arg = jnp.sqrt(3.0) * distance(x1_scaled, x2_scaled)
        exp_part = (1.0 + arg) * jnp.exp(-arg)
        return ((self.scale() ** 2) * exp_part).squeeze()

    def __repr__(self) -> str:
        return "Matern32"


class Matern52(Smooth):
    def call_on_a_pair(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        arg = jnp.sqrt(5.0) * distance(x1_scaled, x2_scaled)
        exp_part = (1.0 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)
        return ((self.scale() ** 2) * exp_part).squeeze()

    def __repr__(self) -> str:
        return "Matern52"


@dataclass
class Polynomial(SubKernel):
    order: float = 1.0
    center: float = 0.0
    center_prior: gd.Distribution = None

    def __post_init__(self):
        super(Polynomial, self).__post_init__()
        self.center = Parameter(self.center, bijector=gb.get_positive_bijector(), prior=self.center_prior)

    def call(self, X1, X2):
        return (X1 @ X2.T + self.center()) ** self.order

    def __get_params__(self):
        return {"center": self.center}

    def set_params(self, params):
        self.center.set(params["center"])

    def __repr__(self) -> str:
        return "Polynomial"
