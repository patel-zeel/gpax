import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors

from stheno import (
    EQ as _EQStheno,
    Matern12 as _Matern12Stheno,
    Matern32 as _Matern32Stheno,
    Matern52 as _Matern52Stheno,
    Linear as _LinearStheno,
)

from typing import List, Union
from jaxtyping import Array
from chex import dataclass
from gpax.base import Base

import lab.jax as B

from plum import dispatch


@dataclass
class Kernel(Base):
    active_dims: List[int] = None
    ARD: bool = True

    def __call__(self, params):
        if self.active_dims is None:
            raise ValueError(
                "active_dims must not be None. It is set automaically on calling _initialise_params() or must be specified manually."
            )
        return self.call(params).select(self.active_dims)

    def initialise_params(self, key, X=None, X_inducing=None):
        if X is not None:
            X = B.uprank(X)
            if self.active_dims is None:
                self.active_dims = list(range(X.shape[1]))
            assert set(self.active_dims) - set(range(X.shape[1])) == set(), "active_dims must be a subset of X.shape[1]"
        if X_inducing is not None:
            X_inducing = B.uprank(X_inducing)
            if self.active_dims is None:
                self.active_dims = list(range(X_inducing.shape[1]))
            assert (
                set(self.active_dims) - set(range(X_inducing.shape[1])) == set()
            ), "active_dims must be a subset of X.shape[1]"
        assert len(set(self.active_dims)) == len(self.active_dims), "active_dims must be unique"
        params = {"kernel": self._identify_and_initiaise_params(key, X, X_inducing)}
        return tree_util.tree_map(lambda x: jnp.asarray(x), params)

    def _identify_and_initiaise_params(self, key, X=None, X_inducing=None):
        if self.__class__.__name__ in ["SumKernel", "ProductKernel"]:
            return self.__initialise_params__(key, X=X, X_inducing=X_inducing)
        elif self.__class__.__name__ == "GibbsKernel":
            return self.__initialise_params__(key, X_inducing=X_inducing)
        else:
            return self.__initialise_params__(key, X=X)

    def get_bijectors(self):
        return {"kernel": self.__get_bijectors__()}

    def __add__(self, other):
        return SumKernel(k1=self, k2=other)

    def __mul__(self, other):
        return ProductKernel(k1=self, k2=other)


@dataclass
class SmoothKernel(Kernel):
    lengthscale: Union[float, List[float], Array] = 1.0
    variance: Union[float, Array] = 1.0

    def call(self, params):
        params = params["kernel"]
        kernel = params["variance"] * self.kernel.stretch(params["lengthscale"])
        return kernel

    def __initialise_params__(self, key, X):
        params = {}
        if self.ARD:
            if self.lengthscale is not None:
                lengthscale = jnp.asarray(self.lengthscale)
                if lengthscale.shape == (len(self.active_dims),):
                    params["lengthscale"] = lengthscale
                elif lengthscale.squeeze().shape == ():
                    params["lengthscale"] = lengthscale.squeeze().repeat(len(self.active_dims))
                else:
                    raise ValueError("lengthscale must be either a scalar or an array of shape (len(active_dims),).")
            else:
                params["lengthscale"] = jnp.ones((len(self.active_dims),))
        else:
            if self.lengthscale is not None:
                lengthscale = jnp.asarray(self.lengthscale)
                if lengthscale.squeeze().shape == ():
                    params["lengthscale"] = lengthscale.squeeze()
                else:
                    raise ValueError("lengthscale must be a scalar when ARD=False.")
            else:
                params["lengthscale"] = jnp.array(1.0)
        if self.variance is not None:
            variance = jnp.asarray(self.variance)
            if variance.squeeze().shape == ():
                params["variance"] = variance.squeeze()
            else:
                raise ValueError("variance must be a scalar.")
        else:
            params["variance"] = jnp.array(1.0)
        return params

    def __get_bijectors__(self):
        return {"lengthscale": tfb.Exp(), "variance": tfb.Exp()}


@dataclass
class RBFKernel(SmoothKernel):
    def __post_init__(self):
        self.kernel = _EQStheno()

    def __repr__(self) -> str:
        return "RBF"


ExpSquaredKernel = RBFKernel
SquaredExpKernel = RBFKernel


@dataclass
class Matern12Kernel(SmoothKernel):
    def __post_init__(self):
        self.kernel = _Matern12Stheno()

    def __repr__(self) -> str:
        return "Matern12"


@dataclass
class Matern32Kernel(SmoothKernel):
    def __post_init__(self):
        self.kernel = _Matern32Stheno()

    def __repr__(self) -> str:
        return "Matern32"


@dataclass
class Matern52Kernel(SmoothKernel):
    def __post_init__(self):
        self.kernel = _Matern52Stheno()

    def __repr__(self) -> str:
        return "Matern52"


@dataclass
class LinearKernel(Kernel):
    variance: Union[float, Array] = 1.0

    def call(self, params):
        params = params["kernel"]
        kernel = params["variance"] * _LinearStheno()
        return kernel

    def __initialise_params__(self, key, X):
        if self.variance is not None:
            return {"variance": self.variance}
        else:
            return {"variance": jnp.array(1.0)}

    def __get_bijectors__(self):
        return {"variance": tfb.Exp()}

    def __repr__(self) -> str:
        return "Linear"


@dataclass
class MathOperationKernel(Kernel):
    k1: Kernel = None
    k2: Kernel = None

    def __post_init__(self):
        assert self.k1 is not None, "k1 must be specified"
        assert self.k2 is not None, "k2 must be specified"

    def call(self, params):
        params = params["kernel"]
        return self.function(self.k1(params["k1"]), self.k2(params["k2"]))

    def __call__(self, params):  # Override to allow for different active_dims
        if self.active_dims is None:
            raise ValueError(
                "active_dims must not be None. It is set automatically on calling initialise_params() or must be specified manually."
            )
        return self.call(params)

    def subkernel_initialise(self, kernel, key, X, X_inducing):
        if kernel.__class__.__name__ in ["SumKernel", "ProductKernel"]:
            return kernel.initialise_params(key, X=X, X_inducing=X_inducing)
        elif kernel.__class__.__name__ == "GibbsKernel":
            return kernel.initialize_params(key, X_inducing=X_inducing)
        else:
            return kernel.initialise_params(key, X=X)

    def __initialise_params__(self, key, X, X_inducing):
        if self.k1.active_dims is None:
            self.k1.active_dims = self.active_dims
        if self.k2.active_dims is None:
            self.k2.active_dims = self.active_dims
        keys = jax.random.split(key, 2)
        return {
            "k1": self.k1.initialise_params(key=keys[0], X=X, X_inducing=X_inducing),
            "k2": self.k2.initialise_params(key=keys[1], X=X, X_inducing=X_inducing),
        }

    def __get_bijectors__(self):
        return {"k1": self.k1.get_bijectors(), "k2": self.k2.get_bijectors()}

    def __repr__(self) -> str:
        return f"({self.k1} {self.operation} {self.k2})"


@dataclass(repr=False)
class ProductKernel(MathOperationKernel):
    def __post_init__(self):
        self.function = lambda k1, k2: k1 * k2
        self.operation = "x"


@dataclass(repr=False)
class SumKernel(MathOperationKernel):
    def __post_init__(self):
        self.function = lambda k1, k2: k1 + k2
        self.operation = "+"
