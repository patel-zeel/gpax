import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

from gpax.bijectors import Identity, Exp

from typing import List, Union
from jaxtyping import Array
from gpax.base import Base
from gpax.utils import squared_distance, distance, get_raw_log_prior
from gpax.distributions import Zero


class Kernel(Base):
    def __init__(self, active_dims=None, ARD=True):
        self.active_dims = active_dims
        self.ARD = ARD

    def __call__(self, params):
        if self.active_dims is None:
            raise ValueError(
                "active_dims must not be None. It is set automatically on calling _initialise_params() or must be specified manually."
            )
        kernel_fn = self.call(params)
        kernel_fn = jax.vmap(kernel_fn, in_axes=(None, 0))
        kernel_fn = jax.vmap(kernel_fn, in_axes=(0, None))
        return self.select(kernel_fn)

    def select(self, kernel_fn):
        def _select(X1, X2):
            # print(X1.shape, X2.shape)
            X1 = X1[:, self.active_dims]
            X2 = X2[:, self.active_dims]
            return kernel_fn(X1, X2)

        return _select

    def initialise_params(self, key, X=None, X_inducing=None):
        if X is not None:
            if self.active_dims is None:
                self.active_dims = list(range(X.shape[1]))
            assert set(self.active_dims) - set(range(X.shape[1])) == set(), "active_dims must be a subset of X.shape[1]"
        if X_inducing is not None:
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

    def get_priors(self):
        return {"kernel": self.__get_priors__()}

    def __add__(self, other):
        return SumKernel(k1=self, k2=other)

    def __mul__(self, other):
        return ProductKernel(k1=self, k2=other)


class SmoothKernel(Kernel):
    def __init__(
        self,
        active_dims=None,
        ARD=True,
        lengthscale=None,
        variance=None,
        lengthscale_prior=Exp()(Zero()),
        variance_prior=Exp()(Zero()),
    ):
        super().__init__(active_dims, ARD)
        self.lengthscale = lengthscale
        self.variance = variance
        self.lengthscale_prior = lengthscale_prior
        self.variance_prior = variance_prior

    def call(self, params):
        params = params["kernel"]
        return self.get_kernel_fn(params)

    def log_prior(self, params, bijectors):
        params = params["kernel"]
        bijectors = bijectors["kernel"]
        return {"kernel": get_raw_log_prior(params, bijectors, self.prior)}

    def __initialise_params__(self, key, X):
        params = {}
        priors = self.__get_priors__()
        keys = jax.random.split(key, 2)
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
                params["lengthscale"] = priors["lengthscale"].sample(
                    seed=keys[0], sample_shape=(len(self.active_dims),)
                )
        else:
            if self.lengthscale is not None:
                lengthscale = jnp.asarray(self.lengthscale)
                if lengthscale.squeeze().shape == ():
                    params["lengthscale"] = lengthscale.squeeze()
                else:
                    raise ValueError("lengthscale must be a scalar when ARD=False.")
            else:
                params["lengthscale"] = priors["lengthscale"].sample(seed=keys[0])
        if self.variance is not None:
            variance = jnp.asarray(self.variance)
            if variance.squeeze().shape == ():
                params["variance"] = variance.squeeze()
            else:
                raise ValueError("variance must be a scalar.")
        else:
            params["variance"] = priors["variance"].sample(seed=keys[1])
        return params

    def __get_bijectors__(self):
        return {"lengthscale": Exp(), "variance": Exp()}

    def __get_priors__(self):
        return {"lengthscale": self.lengthscale_prior, "variance": self.variance_prior}


class RBFKernel(SmoothKernel):
    def get_kernel_fn(self, params):
        def _kernel_fn(X1, X2):
            X1 = X1 / params["lengthscale"]
            X2 = X2 / params["lengthscale"]
            exp_part = jnp.exp(-0.5 * squared_distance(X1, X2))
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "RBF"


ExpSquaredKernel = RBFKernel
SquaredExpKernel = RBFKernel


class Matern12Kernel(SmoothKernel):
    def get_kernel_fn(self, params):
        def _kernel_fn(X1, X2):
            X1 = X1 / params["lengthscale"]
            X2 = X2 / params["lengthscale"]
            exp_part = jnp.exp(-distance(X1, X2))
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Matern12"


class Matern32Kernel(SmoothKernel):
    def get_kernel_fn(self, params):
        def _kernel_fn(X1, X2):
            X1 = X1 / params["lengthscale"]
            X2 = X2 / params["lengthscale"]
            arg = jnp.sqrt(3.0) * distance(X1, X2)
            exp_part = (1.0 + arg) * jnp.exp(-arg)
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Matern32"


class Matern52Kernel(SmoothKernel):
    def get_kernel_fn(self, params):
        def _kernel_fn(X1, X2):
            X1 = X1 / params["lengthscale"]
            X2 = X2 / params["lengthscale"]
            arg = jnp.sqrt(5.0) * distance(X1, X2)
            exp_part = (1 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Matern52"


class PolynomialKernel(SmoothKernel):
    def __init__(self, active_dims=None, ARD=True, lengthscale=1.0, variance=1.0, order=1.0):
        super().__init__(active_dims, ARD, lengthscale, variance)
        self.order = order

    def get_kernel_fn(self, params):
        def _kernel_fn(X1, X2):
            X1 = X1 / params["lengthscale"]
            X2 = X2 / params["lengthscale"]
            return ((X1 @ X2 + params["variance"]) ** self.order).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Polynomial"


class MathOperationKernel(Kernel):
    def __init__(self, k1, k2, active_dims=None, ARD=True):
        super().__init__(active_dims, ARD)
        self.k1 = k1
        self.k2 = k2

    def call(self, params):
        params = params["kernel"]

        def kernel_fn(X1, X2):
            k1 = self.k1.call(params["k1"])(X1, X2)
            k2 = self.k2.call(params["k2"])(X1, X2)
            return self.function(k1, k2)

        return kernel_fn

    def __call__(self, params):  # Override to allow for different active_dims
        if self.active_dims is None:
            raise ValueError(
                "active_dims must not be None. It is set automatically on calling initialise_params() or must be specified manually."
            )
        kernel_fn = self.call(params)
        kernel_fn = jax.vmap(kernel_fn, in_axes=(None, 0))
        kernel_fn = jax.vmap(kernel_fn, in_axes=(0, None))
        return kernel_fn

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

    def __get_priors__(self):
        return {"k1": self.k1.get_priors(), "k2": self.k2.get_priors()}

    def __repr__(self) -> str:
        return f"({self.k1} {self.operation} {self.k2})"


class ProductKernel(MathOperationKernel):
    def __init__(self, k1, k2, active_dims=None, ARD=True):
        super().__init__(k1, k2, active_dims, ARD)
        self.function = lambda k1, k2: k1 * k2
        self.operation = "x"


class SumKernel(MathOperationKernel):
    def __init__(self, k1, k2, active_dims=None, ARD=True):
        super().__init__(k1, k2, active_dims, ARD)
        self.function = lambda k1, k2: k1 + k2
        self.operation = "+"
