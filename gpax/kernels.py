from __future__ import annotations

import jax
import jax.tree_util as jtu
import jax.numpy as jnp

from gpax.core import Parameter, Module
import gpax.distributions as gd
import gpax.bijectors as gb
from gpax.utils import squared_distance, distance, repeat_to_size
from chex import dataclass
from typing import Union


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
    def __call__(self, X1, X2, train_mode=True):
        return self.k1(X1, X2, train_mode) + self.k2(X1, X2, train_mode)


class Product(MathKernel):
    def __call__(self, X1, X2, train_mode=True):
        return self.k1(X1, X2, train_mode) * self.k2(X1, X2, train_mode)


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

    def __call__(self, X1, X2, train_mode=True):
        X1_slice = X1[:, self.active_dims]
        X2_slice = X2[:, self.active_dims]
        return self.call(X1_slice, X2_slice, train_mode)


@dataclass
class Smooth(SubKernel):
    lengthscale: Union[Parameter, float] = 1.0
    scale: Union[Parameter, float] = 1.0

    def __post_init__(self):
        super(Smooth, self).__post_init__()
        if not isinstance(self.lengthscale, Parameter):
            self.lengthscale = Parameter(self.lengthscale, gb.get_positive_bijector())
        if not isinstance(self.scale, Parameter):
            self.scale = Parameter(self.scale, gb.get_positive_bijector())

        raw_value = self.lengthscale.get()
        if self.ARD:
            assert raw_value.size in (
                1,
                self.input_dim,
            ), "lengthscale must be a scalar or an array of shape (input_dim,)."
            raw_value = repeat_to_size(raw_value, self.input_dim)
        else:
            assert raw_value.shape == (), "lengthscale must be a scalar when ARD=False."

        self.lengthscale.set(raw_value)

        assert self.scale.get().shape == (), "scale must be a scalar."

    def call(self, X1, X2, train_mode):
        kernel_fn = self.call_on_a_pair
        kernel_fn = jax.vmap(kernel_fn, in_axes=(None, 0))
        kernel_fn = jax.vmap(kernel_fn, in_axes=(0, None))
        return kernel_fn(X1, X2)

    def __get_params__(self):
        return {"lengthscale": self.lengthscale, "scale": self.scale}

    def set_params(self, raw_params):
        self.lengthscale.set(raw_params["lengthscale"])
        self.scale.set(raw_params["scale"])


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
    center: Union[Parameter, float] = 0.0

    def __post_init__(self):
        super(Polynomial, self).__post_init__()
        if not isinstance(self.center, Parameter):
            self.center = Parameter(self.center, gb.get_positive_bijector())

    def call(self, X1, X2, train_mode):
        return (X1 @ X2.T + self.center()) ** self.order

    def __get_params__(self):
        return {"center": self.center}

    def set_params(self, params):
        self.center.set(params["center"])

    def __repr__(self) -> str:
        return "Polynomial"


@dataclass
class Gibbs(SubKernel):
    method: str = "gp_neurips"
    lengthscale: Union[Parameter, float] = 1.0
    lengthscale_gp: Model = None
    flex_lengthscale: bool = True
    scale: Union[Parameter, float] = 1.0
    scale_gp: Model = None
    flex_scale: bool = True
    X_inducing: Array = None
    # TODO: This supports only 1D lengthscales. ND version is not implemented yet.

    def __post_init__(self):
        super(Gibbs, self).__post_init__()
        std_normal = gd.Normal(loc=0.0, scale=1.0)
        positive_bijector = gb.get_positive_bijector()
        inversed_prior = positive_bijector(gd.Normal(loc=0.0, scale=1.0))

        if not isinstance(self.lengthscale, Parameter):
            if self.flex_lengthscale:
                assert self.X_inducing is not None, "X_inducing must be provided if lengthscale is not a Parameter."
                assert (
                    self.lengthscale_gp is not None
                ), "lengthscale_gp must be provided if lengthscale is not a Parameter."
                ls_bijector = gb.InverseWhite(latent_gp=self.lengthscale_gp, X_inducing=self.X_inducing)
                ls_prior = ls_bijector(std_normal)
                self.lengthscale = Parameter(
                    self.lengthscale, ls_bijector, ls_prior, inversed_init=True, inversed_prior=inversed_prior
                )
            else:
                self.lengthscale = Parameter(self.lengthscale, positive_bijector)

        if not isinstance(self.scale, Parameter):
            if self.flex_scale:
                assert self.X_inducing is not None, "X_inducing must be provided if scale is not a Parameter."
                assert self.scale_gp is not None, "scale_gp must be provided if scale is not a Parameter."
                s_bijector = gb.InverseWhite(latent_gp=self.scale_gp, X_inducing=self.X_inducing)
                s_prior = s_bijector(std_normal)
                self.scale = Parameter(
                    self.scale, s_bijector, s_prior, inversed_init=True, inversed_prior=inversed_prior
                )
            else:
                self.scale = Parameter(self.scale, positive_bijector)

    def __get_params__(self):
        params = {"lengthscale": self.lengthscale, "scale": self.scale}
        if self.flex_lengthscale:
            params["lengthscale_gp"] = self.lengthscale._bijector.latent_gp.__get_params__()
            params["X_inducing"] = self.lengthscale._bijector.X_inducing
        if self.flex_scale:
            params["scale_gp"] = self.scale._bijector.latent_gp.__get_params__()
            params["X_inducing"] = self.scale._bijector.X_inducing

        return params

    def set_params(self, raw_params):
        self.lengthscale.set(raw_params["lengthscale"])
        self.scale.set(raw_params["scale"])
        if self.flex_lengthscale:
            self.lengthscale._bijector.latent_gp.set_params(raw_params["lengthscale_gp"])
            self.lengthscale._bijector.X_inducing.set(raw_params["X_inducing"])

        if self.flex_scale:
            self.scale._bijector.latent_gp.set_params(raw_params["scale_gp"])
            self.scale._bijector.X_inducing.set(raw_params["X_inducing"])

    def call(self, X1, X2, train_mode=True):
        dist_f = jax.vmap(squared_distance, in_axes=(None, 0))
        dist_f = jax.vmap(dist_f, in_axes=(0, None))

        if self.flex_lengthscale:
            positive_bijector = gb.get_positive_bijector()
            X_inducing = self.lengthscale._bijector.X_inducing()
            ls_inducing = self.lengthscale()
            if self.method == "gp_neurips":
                train_mode = False
            if train_mode:
                ls1 = ls_inducing
                ls2 = ls1
            else:
                raw_ls_inducing = positive_bijector.inverse(ls_inducing)
                ls1, ls2 = jtu.tree_map(
                    lambda x: positive_bijector(
                        self.lengthscale._bijector.latent_gp.predict(
                            X_inducing,
                            raw_ls_inducing,
                            x,
                            include_noise=False,
                            return_cov=False,
                            include_train_likelihood=False,
                        )
                    ),
                    (X1, X2),
                )
            ls1, ls2 = ls1.reshape(-1, 1), ls2.reshape(-1, 1)  # 1D only

            l_avg_square = (ls1**2 + ls2.T**2) / 2.0
            prefix_part = jnp.sqrt(ls1 * ls2.T / l_avg_square)

            squared_dist = dist_f(X1, X2)

            exp_part = jnp.exp(-squared_dist / (2.0 * l_avg_square))
        else:
            prefix_part = 1.0
            exp_part = jnp.exp(-dist_f(X1, X2) / (2.0 * self.lengthscale() ** 2))

        if self.flex_scale:
            positive_bijector = gb.get_positive_bijector()
            X_inducing = self.scale._bijector.X_inducing()
            s_inducing = self.scale()
            if self.method == "gp_neurips":
                train_mode = False

            if train_mode:
                s1 = s_inducing
                s2 = s1
            else:
                raw_s_inducing = positive_bijector.inverse(s_inducing)
                s1, s2 = jtu.tree_map(
                    lambda x: positive_bijector(
                        self.scale._bijector.latent_gp.predict(
                            X_inducing,
                            raw_s_inducing,
                            x,
                            include_noise=False,
                            return_cov=False,
                            include_train_likelihood=False,
                        )
                    ),
                    (X1, X2),
                )
            s1, s2 = s1.reshape(-1, 1), s2.reshape(-1, 1)  # 1D only
            variance = s1 * s2.T
        else:
            variance = self.scale() ** 2

        return variance * prefix_part * exp_part
