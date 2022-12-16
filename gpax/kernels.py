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
        X1_slice, X2_slice = self.slice_inputs(X1, X2)
        if self.training:
            return kernel_fn(X1_slice, X2_slice), 0.0
        else:
            return kernel_fn(X1_slice, X2_slice)


class RBF(Smooth):
    def pair_wise(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        sqr_dist = squared_distance(x1_scaled, x2_scaled)
        return ((self.scale() ** 2) * jnp.exp(-0.5 * sqr_dist)).squeeze()

    def __repr__(self) -> str:
        return "RBF"


ExpSquared = RBF
SquaredExp = RBF


class Matern12(Smooth):
    def pair_wise(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        dist = distance(x1_scaled, x2_scaled)
        return ((self.scale() ** 2) * jnp.exp(-dist)).squeeze()

    def __repr__(self) -> str:
        return "Matern12"


class Matern32(Smooth):
    def pair_wise(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        arg = jnp.sqrt(3.0) * distance(x1_scaled, x2_scaled)
        exp_part = (1.0 + arg) * jnp.exp(-arg)
        return ((self.scale() ** 2) * exp_part).squeeze()

    def __repr__(self) -> str:
        return "Matern32"


class Matern52(Smooth):
    def pair_wise(self, x1, x2):
        x1_scaled = x1 / self.lengthscale()
        x2_scaled = x2 / self.lengthscale()
        arg = jnp.sqrt(5.0) * distance(x1_scaled, x2_scaled)
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
        X1_slice, X2_slice = self.slice_inputs(X1, X2)
        if self.training:
            return (X1_slice @ X2_slice.T + self.center()) ** self.order, 0.0
        else:
            return (X1_slice @ X2_slice.T + self.center()) ** self.order

    def __repr__(self) -> str:
        return "Polynomial"


class Gibbs(Kernel):
    def __init__(
        self,
        flex_lengthscale: bool,
        flex_scale: bool,
        lengthscale: float,
        scale: float,
        X: Array,
        active_dims: list,
        ARD: bool,
    ):
        super(Gibbs, self).__init__(X, active_dims)
        self.flex_lengthscale = flex_lengthscale
        self.flex_scale = flex_scale
        self.ARD = ARD

        if self.flex_lengthscale is False:
            if self.ARD:
                lengthscale = repeat_to_size(lengthscale, X.shape[1])
            else:
                assert lengthscale.shape == (), "lengthscale must be a scalar when ARD=False."
            self.lengthscale = Parameter(lengthscale, get_positive_bijector())

        if self.flex_scale is False:
            assert scale.shape == (), "scale must be a scalar."
            self.scale = Parameter(scale, get_positive_bijector())

    @staticmethod
    def per_dim_std_cov(X1, X2, ls1, ls2):
        dist_f = jax.vmap(squared_distance, in_axes=(None, 0))
        dist_f = jax.vmap(dist_f, in_axes=(0, None))

        ls1, ls2 = ls1.reshape(-1, 1), ls2.reshape(-1, 1)  # 1D only

        l_avg_square = (ls1**2 + ls2.T**2) / 2.0
        prefix_part = jnp.sqrt(ls1 * ls2.T / l_avg_square)

        squared_dist = dist_f(X1, X2)

        exp_part = jnp.exp(-squared_dist / (2.0 * l_avg_square))
        return prefix_part * exp_part

    def get_kernel_fn(self, X_inducing: Array = None):
        def kernel_fn(X1, X2):
            X1_slice, X2_slice, X_inducing_slice = self.slice_inputs(X1, X2, X_inducing)
            ls_gp_fn = self.ls_gp(X_inducing_slice)
            scale_gp_fn = self.scale_gp(X_inducing_slice)

            #### Lengthscale ####
            if self.flex_lengthscale:
                if self.training:
                    # X1 and X2 should be the same here
                    ls1, log_ls_prior = ls_gp_fn(X1_slice)
                    ls2 = ls1
                else:
                    ls1, ls2 = ls_gp_fn(X1_slice), ls_gp_fn(X2_slice)
            else:
                ls1 = self.lengthscale().reshape(1, -1).repeat(X1_slice.shape[0], axis=0).squeeze()
                ls2 = ls1 = jnp.atleast_1d(ls1)

            if self.ARD:
                # print(X1_slice.shape, X2_slice.shape, ls1.shape, ls2.shape)
                std_cov = jax.vmap(self.per_dim_std_cov, in_axes=(1, 1, 1, 1), out_axes=2)(
                    X1_slice, X2_slice, ls1, ls2
                ).prod(axis=2)
            else:
                std_cov = self.per_dim_std_cov(X1_slice, X2_slice, ls1, ls2)

            #### Scale ####
            if self.flex_scale:
                if self.training:
                    s1, log_s_prior = scale_gp_fn(X1_slice)
                    s2 = s1
                else:
                    s1, s2 = scale_gp_fn(X1_slice), scale_gp_fn(X2_slice)
            else:
                s1 = self.scale().reshape(1, 1).repeat(X1_slice.shape[0], axis=0).squeeze()
                s2 = s1 = jnp.atleast_1d(s1)

            s1, s2 = s1.reshape(-1, 1), s2.reshape(-1, 1)  # 1D only
            variance = s1 * s2.T

            if self.training:
                return variance * std_cov, log_ls_prior + log_s_prior
            else:
                return variance * std_cov

        return kernel_fn


class GibbsHeinonen(Gibbs):
    def __init__(
        self,
        flex_lengthscale: bool = True,
        flex_scale: bool = True,
        lengthscale: float = 1.0,
        scale: float = 1.0,
        latent_lengthscale_ell: float = 1.0,
        latent_scale_ell: float = 1.0,
        latent_lengthscale_sigma: float = 1.0,
        latent_scale_sigma: float = 1.0,
        latent_kernel_type: Kernel = RBF,
        X_inducing: Array = None,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super(GibbsHeinonen, self).__init__(
            flex_lengthscale, flex_scale, lengthscale, scale, X_inducing, active_dims, ARD
        )

        if self.flex_lengthscale:
            self.ls_gp = LatentGPHeinonen(
                X_inducing, latent_lengthscale_ell, latent_lengthscale_sigma, latent_kernel_type, vmap=ARD
            )
        if self.flex_scale:
            self.scale_gp = LatentGPHeinonen(
                X_inducing, latent_scale_ell, latent_scale_sigma, latent_kernel_type, vmap=False
            )


class GibbsDeltaInducing(Gibbs):
    def __init__(
        self,
        flex_lengthscale: bool,
        flex_scale: bool,
        lengthscale: float = 1,
        scale: float = 1,
        latent_lengthscale_ell: float = 1.0,
        latent_scale_ell: float = 1.0,
        latent_lengthscale_sigma: float = 1.0,
        latent_scale_sigma: float = 1.0,
        latent_kernel_type: Kernel = RBF,
        X_inducing: Array = None,
        active_dims: list = None,
        ARD: bool = True,
    ):
        super().__init__(flex_lengthscale, flex_scale, lengthscale, scale, X_inducing, active_dims, ARD)

        if self.flex_lengthscale:
            self.ls_gp = LatentGPDeltaInducing(
                X_inducing, latent_lengthscale_ell, latent_lengthscale_sigma, latent_kernel_type, vmap=ARD
            )
        if self.flex_scale:
            self.scale_gp = LatentGPDeltaInducing(
                X_inducing, latent_scale_ell, latent_scale_sigma, latent_kernel_type, vmap=False
            )
