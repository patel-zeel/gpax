from __future__ import annotations

import jax
import jax.tree_util as jtu
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from gpax.core import Parameter, Module, get_positive_bijector, get_default_jitter
from gpax.models import LatentGPHeinonen, LatentGPDeltaInducing
from gpax.utils import squared_distance, distance, repeat_to_size

from jaxtyping import Array, Float
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpax.models import LatentModel


class MetaKernel(Module):
    def __add__(self, other):
        return Sum(k1=self, k2=other)

    def __mul__(self, other):
        if isinstance(other, MetaKernel):
            return Product(k1=self, k2=other)
        else:
            other = jnp.asarray(other)
            X = jnp.ones((1, 1))  # dummy
            return Scale(X, base_kernel=self, variance=other)

    def __rmul__(self, other):
        return self.__mul__(other)


class MathKernel(MetaKernel):
    def __init__(self, k1: MetaKernel, k2: MetaKernel):
        super(MathKernel, self).__init__()
        self.k1 = k1
        self.k2 = k2

    def get_kernel_fn(self, X_inducing: Parameter = None):
        k1 = self.k1.get_kernel_fn(X_inducing=X_inducing)
        k2 = self.k2.get_kernel_fn(X_inducing=X_inducing)

        def kernel_fn(X1, X2):
            if self._training:
                K1, log_prior1 = k1(X1, X2)
                K2, log_prior2 = k2(X1, X2)
                return self.operation(K1, K2), log_prior1 + log_prior2  # return log_prior
            else:
                return self.operation(k1(X1, X2), k2(X1, X2))

        return kernel_fn


class Sum(MathKernel):
    @staticmethod
    def operation(K1, K2):
        return K1 + K2

    def __repr__(self) -> str:
        return f"({self.k1} + {self.k2})"


class Product(MathKernel):
    @staticmethod
    def operation(K1, K2):
        return K1 * K2

    def __repr__(self) -> str:
        return f"({self.k1} * {self.k2})"


class Kernel(MetaKernel):
    def __init__(self, X: Array, active_dims: list):
        super(Kernel, self).__init__()
        self.active_dims = active_dims
        self._num_dim = X.shape[1]
        self._num_data = X.shape[0]
        if self.active_dims is None:
            assert X is not None, "X must be specified if active_dims is None"
            self.active_dims = list(range(X.shape[1]))
        else:
            self.active_dims = active_dims

        # Sanity check
        # print(self.active_dims)
        assert len(set(self.active_dims)) == len(self.active_dims), "active_dims must be unique."
        assert isinstance(sum(self.active_dims), int), "active_dims must be a list of integers."
        assert len(self.active_dims) <= self._num_dim, "active_dims can not be larger than the input dimensions."
        assert max(self.active_dims) < self._num_dim, "active_dims must be a subset of the input dimensions."

    def slice_inputs(self, *args):
        return jtu.tree_map(lambda x: x[:, self.active_dims], args)


class Scale(Kernel):
    def __init__(self, X: Array, base_kernel: MetaKernel, variance: float = 1.0, active_dims: list = None):
        super(Scale, self).__init__(X, active_dims)
        self.base_kernel = base_kernel

        variance = jnp.asarray(variance)
        assert variance.shape == (), "variance must be a scalar."
        self.variance = Parameter(variance, get_positive_bijector())

    def get_kernel_fn(self, X_inducing: Array = None):
        def kernel_fn(X1, X2):
            if self._training:
                K, log_prior = self.base_kernel.get_kernel_fn(X_inducing)(X1, X2)
                return self.variance() * K, log_prior
            else:
                K = self.base_kernel.get_kernel_fn(X_inducing)(X1, X2)
                return self.variance() * K

        return kernel_fn

    def __repr__(self) -> str:
        return f"Scale({self.base_kernel})"


class Smooth(Kernel):
    def __init__(
        self,
        X: Array,
        lengthscale: float = 1.0,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super(Smooth, self).__init__(X, active_dims)
        lengthscale = jnp.asarray(lengthscale)
        self.ARD = ARD

        if self.ARD:
            assert lengthscale.size in (
                1,
                len(self.active_dims),
            ), "lengthscale must be a scalar or an array of shape (input_dim,)."
            lengthscale = repeat_to_size(lengthscale, len(self.active_dims))
        else:
            assert lengthscale.shape == (), "lengthscale must be a scalar when ARD=False."

        self.lengthscale = Parameter(lengthscale, get_positive_bijector())

    def get_kernel_fn(self, X_inducing: Parameter = None):
        return self.call

    def call(self, X1, X2):
        kernel_fn = self.pair_wise
        kernel_fn = jax.vmap(kernel_fn, in_axes=(None, 0))
        kernel_fn = jax.vmap(kernel_fn, in_axes=(0, None))
        X1, X2 = self.slice_inputs(X1, X2)
        if self._training:
            return kernel_fn(X1, X2), 0.0
        else:
            return kernel_fn(X1, X2)


class Wind(Smooth):
    def pair_wise(self, x1, x2):
        lat1, lon1, deg1 = x1[0], x1[1], x1[2]
        lat2, lon2, deg2 = x2[0], x2[1], x2[2]
        rad1 = self.degree_to_rad(deg1)
        rad2 = self.degree_to_rad(deg2)
        return jnp.exp(-jnp.square((lat1 - lat2) * jnp.sin(rad1) - jnp.cos(rad2) * (lon1 - lon2))).squeeze()

    def degree_to_rad(self, deg):
        return deg * jnp.pi / 180


class RBF(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        sqr_dist = squared_distance(x1, x2)
        return jnp.exp(-0.5 * sqr_dist).squeeze()

    def __repr__(self) -> str:
        return "RBF"


ExpSquared = RBF
SquaredExp = RBF


class Matern12(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        dist = distance(x1, x2)
        return jnp.exp(-dist).squeeze()

    def __repr__(self) -> str:
        return "Matern12"


Exponential = Matern12


class Matern32(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        arg = jnp.sqrt(3.0) * distance(x1, x2)
        exp_part = (1.0 + arg) * jnp.exp(-arg)
        return exp_part.squeeze()

    def __repr__(self) -> str:
        return "Matern32"


class Matern52(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        arg = jnp.sqrt(5.0) * distance(x1, x2)
        exp_part = (1.0 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)
        return exp_part.squeeze()

    def __repr__(self) -> str:
        return "Matern52"


class Hamming(Smooth):
    """
    This is a 1D kernel. Do not use it for multidimensional inputs.
    """

    def pair_wise(self, x1, x2):
        hamming_distance = (x1 != x2) / self.lengthscale()
        exp_part = jnp.exp(-hamming_distance)
        return exp_part.squeeze()

    def __repr__(self) -> str:
        return "Hamming"


class RationalQuadratic(Smooth):
    def __init__(
        self,
        X: Array,
        lengthscale: float = 1.0,
        alpha: float = 1.0,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super(RationalQuadratic, self).__init__(X, lengthscale, active_dims, ARD)
        self.alpha = Parameter(alpha, get_positive_bijector())

    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        arg = squared_distance(x1, x2)
        return (1.0 + arg / 2.0) ** (-self.alpha())

    def __repr__(self) -> str:
        return "RationalQuadratic"


class Periodic(Smooth):
    def __init__(
        self,
        X: Array,
        lengthscale: float = 1.0,
        period: float = 1.0,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super(Periodic, self).__init__(X, lengthscale, active_dims, ARD)
        assert len(self.active_dims) == 1, "Periodic kernel only supports 1D inputs."
        self.period = Parameter(period, get_positive_bijector())

    def pair_wise(self, x1, x2):
        # arg = jnp.sin(jnp.pi * distance(x1, x2) / self.period())
        # return jnp.exp(-0.5 * jnp.square(arg / self.lengthscale())).squeeze()

        sine_squared = (jnp.sin(jnp.pi * (x1 - x2) / self.period()) / self.lengthscale()) ** 2
        return jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))

    def __repr__(self) -> str:
        return "Periodic"


class Polynomial(Kernel):
    def __init__(self, order: float = 1.0, center: float = 0.0, X: Array = None, active_dims: list = None):
        super(Polynomial, self).__init__(X, active_dims)
        self.order = order
        self.center = Parameter(center, get_positive_bijector())

    def get_kernel_fn(self, X_inducing: Array = None):
        return self.call

    def call(self, X1, X2):
        X1, X2 = self.slice_inputs(X1, X2)
        if self._training:
            return (X1 @ X2.T + self.center()) ** self.order, 0.0
        else:
            return (X1 @ X2.T + self.center()) ** self.order

    def __repr__(self) -> str:
        return "Polynomial"


class InputDependentScale(Kernel):
    def __init__(
        self,
        X_inducing: Array,
        base_kernel: MetaKernel,
        latent_model: LatentModel = None,
        active_dims: list = None,
    ):
        super(InputDependentScale, self).__init__(X_inducing, active_dims)
        self.base_kernel = base_kernel

        self.latent_model = latent_model
        assert self.latent_model.vmap is False, "latent_model must be a non-vmap model."

    def get_kernel_fn(self, X_inducing: Array):
        def kernel_fn(X1, X2, X_inducing):  # X_inducing is passed here to fix local variable error
            if self._training:
                K, log_kernel_prior = self.base_kernel.get_kernel_fn(X_inducing)(X1, X2)
            else:
                K = self.base_kernel.get_kernel_fn(X_inducing)(X1, X2)

            X1, X2, X_inducing = self.slice_inputs(X1, X2, X_inducing)
            sigma_fn = self.latent_model(X_inducing)
            if self._training:
                sigma1, log_sigma_prior = sigma_fn(X1)
                sigma2 = sigma1
            else:
                sigma1, sigma2 = sigma_fn(X1), sigma_fn(X2)
            sigma1, sigma2 = sigma1.reshape(-1, 1), sigma2.reshape(-1, 1)  # 1D only
            variance = sigma1 * sigma2.T

            if self._training:
                return variance * K, log_kernel_prior + log_sigma_prior
            else:
                return variance * K

        return lambda X1, X2: kernel_fn(X1, X2, X_inducing)

    def __repr__(self) -> str:
        return f"InputDependentScale({self.base_kernel})"


class Gibbs(Kernel):
    def __init__(
        self,
        X_inducing: Array,
        latent_model: LatentModel,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super(Gibbs, self).__init__(X_inducing, active_dims)
        self.ARD = ARD

        self.latent_model = latent_model
        # assert self.latent_model.vmap is True, "latent_model must be a vmap model."

    @staticmethod
    def per_dim_std_cov(X1, X2, ell1, ell2):
        dist_f = jax.vmap(squared_distance, in_axes=(None, 0))
        dist_f = jax.vmap(dist_f, in_axes=(0, None))

        ell1, ell2 = ell1.reshape(-1, 1), ell2.reshape(-1, 1)  # 1D only

        ell_avg_square = (ell1**2 + ell2.T**2) / 2.0
        prefix_part = jnp.sqrt(ell1 * ell2.T / ell_avg_square)

        squared_dist = dist_f(X1, X2)

        exp_part = jnp.exp(-squared_dist / (2.0 * ell_avg_square))
        return prefix_part * exp_part

    def get_kernel_fn(self, X_inducing: Array):
        def kernel_fn(X1, X2, X_inducing):  # X_inducing is passed here so that Local Variable error is not raised
            X1, X2, X_inducing = self.slice_inputs(X1, X2, X_inducing)

            ell_fn = self.latent_model(X_inducing)
            if self._training:
                # X1 and X2 should be the same here
                ell1, log_ell_prior = ell_fn(X1)

                ell2 = ell1
            else:
                ell1, ell2 = ell_fn(X1), ell_fn(X2)

            if self.ARD:
                std_cov = 1.0
                # Not applying a vmap here because it goes out of memory
                for i in range(X1.shape[1]):
                    print("Z-debug", X1.shape, X2.shape, ell1.shape, ell2.shape)
                    std_cov *= self.per_dim_std_cov(X1[:, i], X2[:, i], ell1[:, i], ell2[:, i])
            else:
                std_cov = self.per_dim_std_cov(X1, X2, ell1, ell2)

            if self._training:
                # print(f"{log_ell_prior.sum()=:.20f}, {log_sigma_prior=:.20f}", end=" ")  # DEBUG
                return std_cov, log_ell_prior.sum()
            else:
                return std_cov

        return lambda X1, X2: kernel_fn(X1, X2, X_inducing)

    def __repr__(self) -> str:
        return f"Gibbs"
