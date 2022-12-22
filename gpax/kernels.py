from __future__ import annotations

import jax
import jax.tree_util as jtu
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from gpax.core import Parameter, Module, get_positive_bijector
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
        return Product(k1=self, k2=other)


class MathKernel(MetaKernel):
    def __init__(self, k1: MetaKernel, k2: MetaKernel):
        super(MathKernel, self).__init__()
        self.k1 = k1
        self.k2 = k2

    def get_kernel_fn(self, X_inducing: Parameter = None):
        k1 = self.k1.get_kernel_fn(X_inducing=X_inducing)
        k2 = self.k2.get_kernel_fn(X_inducing=X_inducing)

        def kernel_fn(X1, X2):
            if self.training:
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


class Product(MathKernel):
    @staticmethod
    def operation(K1, K2):
        return K1 * K2


class Kernel(MetaKernel):
    def __init__(self, X: Array, active_dims: list):
        super(Kernel, self).__init__()
        self.active_dims = active_dims
        if self.active_dims is None:
            assert X is not None, "X must be specified if active_dims is None"
            self.active_dims = list(range(X.shape[1]))
        else:
            self.active_dims = active_dims

    def slice_inputs(self, *args):
        return jtu.tree_map(lambda x: x[:, self.active_dims], args)


class Smooth(Kernel):
    def __init__(
        self,
        X: Array,
        lengthscale: float = 1.0,
        scale: float = 1.0,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super(Smooth, self).__init__(X, active_dims)
        lengthscale = jnp.asarray(lengthscale)
        scale = jnp.asarray(scale)
        self.ARD = ARD

        if self.ARD:
            assert lengthscale.size in (
                1,
                X.shape[1],
            ), "lengthscale must be a scalar or an array of shape (input_dim,)."
            lengthscale = repeat_to_size(lengthscale, X.shape[1])
        else:
            assert lengthscale.shape == (), "lengthscale must be a scalar when ARD=False."

        assert scale.shape == (), "scale must be a scalar."

        self.lengthscale = Parameter(lengthscale, get_positive_bijector())
        self.scale = Parameter(scale, get_positive_bijector())

    def get_kernel_fn(self, X_inducing: Parameter = None):
        return self.call

    def call(self, X1, X2):
        kernel_fn = self.pair_wise
        kernel_fn = jax.vmap(kernel_fn, in_axes=(None, 0))
        kernel_fn = jax.vmap(kernel_fn, in_axes=(0, None))
        X1, X2 = self.slice_inputs(X1, X2)
        if self.training:
            return kernel_fn(X1, X2), 0.0
        else:
            return kernel_fn(X1, X2)


class RBF(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        sqr_dist = squared_distance(x1, x2)
        return ((self.scale() ** 2) * jnp.exp(-0.5 * sqr_dist)).squeeze()

    def __repr__(self) -> str:
        return "RBF"


ExpSquared = RBF
SquaredExp = RBF


class Matern12(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        dist = distance(x1, x2)
        return ((self.scale() ** 2) * jnp.exp(-dist)).squeeze()

    def __repr__(self) -> str:
        return "Matern12"


class Matern32(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        arg = jnp.sqrt(3.0) * distance(x1, x2)
        exp_part = (1.0 + arg) * jnp.exp(-arg)
        return ((self.scale() ** 2) * exp_part).squeeze()

    def __repr__(self) -> str:
        return "Matern32"


class Matern52(Smooth):
    def pair_wise(self, x1, x2):
        x1 = x1 / self.lengthscale()
        x2 = x2 / self.lengthscale()
        arg = jnp.sqrt(5.0) * distance(x1, x2)
        exp_part = (1.0 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)
        return ((self.scale() ** 2) * exp_part).squeeze()

    def __repr__(self) -> str:
        return "Matern52"


class Polynomial(Kernel):
    def __init__(self, order: float = 1.0, center: float = 0.0, X: Array = None, active_dims: list = None):
        super(Polynomial, self).__init__(X, active_dims)
        self.order = order
        self.center = Parameter(center, get_positive_bijector())

    def get_kernel_fn(self, X_inducing: Parameter = None):
        return self.call

    def call(self, X1, X2):
        X1, X2 = self.slice_inputs(X1, X2)
        if self.training:
            return (X1 @ X2.T + self.center()) ** self.order, 0.0
        else:
            return (X1 @ X2.T + self.center()) ** self.order

    def __repr__(self) -> str:
        return "Polynomial"


class Gibbs(Kernel):
    def __init__(
        self,
        X_inducing: Array = None,
        ell_model: LatentModel = None,
        sigma_model: LatentModel = None,
        flex_ell: bool = True,
        flex_sigma: bool = True,
        lengthscale: float = 1.0,
        scale: float = 1.0,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super(Gibbs, self).__init__(X_inducing, active_dims)
        self.flex_ell = flex_ell
        self.flex_sigma = flex_sigma
        self.ARD = ARD

        if self.flex_ell:
            self.ell_model = ell_model
            assert self.ell_model.vmap is True, "ell_model must be a vmap model."
        else:
            if self.ARD:
                lengthscale = repeat_to_size(lengthscale, X_inducing.shape[1])
            else:
                assert lengthscale.shape == (), "lengthscale must be a scalar when ARD=False."
            self.lengthscale = Parameter(lengthscale, get_positive_bijector())

        if self.flex_sigma:
            self.sigma_model = sigma_model
            assert self.sigma_model.vmap is False, "sigma_model must be a non-vmap model."

        if self.flex_sigma is False:
            assert scale.shape == (), "scale must be a scalar."
            self.scale = Parameter(scale, get_positive_bijector())

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

    def get_kernel_fn(self, X_inducing: Array = None):
        def kernel_fn(X1, X2, X_inducing):
            X1, X2, X_inducing = self.slice_inputs(X1, X2, X_inducing)

            ell_fn = self.ell_model(X_inducing)
            sigma_fn = self.sigma_model(X_inducing)

            #### Lengthscale ####
            if self.flex_ell:
                if self.training:
                    # X1 and X2 should be the same here
                    ell1, log_ell_prior = ell_fn(X1)

                    ell2 = ell1
                else:
                    ell1, ell2 = ell_fn(X1), ell_fn(X2)
            else:
                ell1 = self.lengthscale().reshape(1, -1).repeat(X1.shape[0], axis=0).squeeze()
                ell2 = ell1 = jnp.atleast_1d(ell1)

            if self.ARD:

                std_cov = jax.vmap(self.per_dim_std_cov, in_axes=(1, 1, 1, 1), out_axes=2)(X1, X2, ell1, ell2).prod(
                    axis=2
                )
            else:
                std_cov = self.per_dim_std_cov(X1, X2, ell1, ell2)

            #### Scale ####
            if self.flex_sigma:
                if self.training:
                    sigma1, log_sigma_prior = sigma_fn(X1)
                    sigma2 = sigma1
                else:
                    sigma1, sigma2 = sigma_fn(X1), sigma_fn(X2)
            else:
                sigma1 = self.scale().reshape(1, 1).repeat(X1.shape[0], axis=0).squeeze()
                sigma2 = sigma1 = jnp.atleast_1d(sigma1)

            sigma1, sigma2 = sigma1.reshape(-1, 1), sigma2.reshape(-1, 1)  # 1D only
            variance = sigma1 * sigma2.T

            if self.training:
                # print(f"{log_ell_prior.sum()=:.20f}, {log_sigma_prior=:.20f}", end=" ")  # DEBUG
                return variance * std_cov, log_ell_prior.sum() + log_sigma_prior
            else:
                return variance * std_cov

        return lambda X1, X2: kernel_fn(X1, X2, X_inducing)
