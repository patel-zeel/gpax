import jax
import jax.numpy as jnp

from gpax.core import Base, get_raw_log_prior, get_positive_bijector
from gpax.utils import squared_distance, distance


class Kernel(Base):
    def select(self, kernel_fn):
        def _select(X1, X2):
            assert self.active_dims is not None, "active_dims must not be None"
            X1 = X1[:, self.active_dims]
            X2 = X2[:, self.active_dims]
            return kernel_fn(X1, X2)

        return _select

    def set_active_dims(self, aux):
        if self.active_dims is not None:
            pass
        else:
            if "X" in aux:
                self.active_dims = list(range(aux["X"].shape[1]))
            elif "X_inducing" in aux:
                self.active_dims = list(range(aux["X_inducing"].shape[1]))
            else:
                raise ValueError("aux must contain X or X_inducing")

    def __call__(self, params, aux=None, prior_type=None):
        kernel_fn = self.call(params, aux, prior_type)
        if isinstance(self, MathOperation):
            return kernel_fn
        else:
            return self.select(kernel_fn)

    def __add__(self, other):
        return Sum(k1=self, k2=other)

    def __mul__(self, other):
        return Product(k1=self, k2=other)


class Smooth(Kernel):
    def __init__(
        self,
        active_dims=None,
        ARD=True,
        lengthscale=1.0,
        variance=1.0,
        lengthscale_prior=None,
        variance_prior=None,
    ):
        self.active_dims = active_dims
        self.ARD = ARD
        self.lengthscale = lengthscale
        self.variance = variance
        self.lengthscale_prior = lengthscale_prior
        self.variance_prior = variance_prior

    def call(self, params, aux=None, prior_type=None):
        kernel_fn = self.get_kernel_fn(params, aux)
        kernel_fn = jax.vmap(kernel_fn, in_axes=(None, 0))
        kernel_fn = jax.vmap(kernel_fn, in_axes=(0, None))

        if prior_type is None:
            return kernel_fn
        else:
            # For Smooth kernels, prior is not applicable within kernel, so return zero.
            log_prior = jnp.zeros(())
            return lambda X1, X2: kernel_fn(X1, X2), log_prior

    def log_prior(self, params):
        return get_raw_log_prior(params, self.constraints, self.priors)

    def __initialize_params__(self, aux):
        self.set_active_dims(aux)
        lengthscale = jnp.asarray(self.lengthscale)
        variance = jnp.asarray(self.variance)

        params = {}
        if self.ARD:
            if lengthscale.shape == (len(self.active_dims),):
                params["lengthscale"] = lengthscale
            elif lengthscale.squeeze().shape == ():
                params["lengthscale"] = lengthscale.squeeze().repeat(len(self.active_dims))
            else:
                raise ValueError("lengthscale must be either a scalar or an array of shape (len(active_dims),).")
        else:
            if lengthscale.squeeze().shape == ():
                params["lengthscale"] = lengthscale.squeeze()
            else:
                raise ValueError("lengthscale must be a scalar when ARD=False.")

        if variance.squeeze().shape == ():
            params["variance"] = variance.squeeze()
        else:
            raise ValueError("variance must be a scalar.")

        self.constraints = {"lengthscale": get_positive_bijector(), "variance": get_positive_bijector()}
        self.priors = {"lengthscale": self.lengthscale_prior, "variance": self.variance_prior}

        return params


class RBF(Smooth):
    def get_kernel_fn(self, params, aux):
        def _kernel_fn(X1, X2):
            X1_scaled = X1 / params["lengthscale"]
            X2_scaled = X2 / params["lengthscale"]
            exp_part = jnp.exp(-0.5 * squared_distance(X1_scaled, X2_scaled))
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "RBF"


ExpSquared = RBF
SquaredExp = RBF


class Matern12(Smooth):
    def get_kernel_fn(self, params, aux):
        def _kernel_fn(X1, X2):
            X1_scaled = X1 / params["lengthscale"]
            X2_scaled = X2 / params["lengthscale"]
            exp_part = jnp.exp(-distance(X1_scaled, X2_scaled))
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Matern12"


class Matern32(Smooth):
    def get_kernel_fn(self, params, aux):
        def _kernel_fn(X1, X2):
            X1_scaled = X1 / params["lengthscale"]
            X2_scaled = X2 / params["lengthscale"]
            arg = jnp.sqrt(3.0) * distance(X1_scaled, X2_scaled)
            exp_part = (1.0 + arg) * jnp.exp(-arg)
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Matern32"


class Matern52(Smooth):
    def get_kernel_fn(self, params, aux):
        def _kernel_fn(X1, X2):
            X1_scaled = X1 / params["lengthscale"]
            X2_scaled = X2 / params["lengthscale"]
            arg = jnp.sqrt(5.0) * distance(X1_scaled, X2_scaled)
            exp_part = (1 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)
            return (params["variance"] * exp_part).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Matern52"


class Polynomial(Smooth):
    def __init__(
        self,
        active_dims=None,
        ARD=True,
        lengthscale=1.0,
        variance=1.0,
        order=1.0,
        lengthscale_prior=None,
        variance_prior=None,
    ):
        super().__init__(active_dims, ARD, lengthscale, variance, lengthscale_prior, variance_prior)
        self.order = order

    def get_kernel_fn(self, params, aux):
        def _kernel_fn(X1, X2):
            X1_scaled = X1 / params["lengthscale"]
            X2_scaled = X2 / params["lengthscale"]
            return ((X1_scaled @ X2_scaled + params["variance"]) ** self.order).squeeze()

        return _kernel_fn

    def __repr__(self) -> str:
        return "Polynomial"


class MathOperation(Kernel):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def call(self, params, prior_type=None, aux=None):
        def kernel_fn(X1, X2):
            k1 = self.k1.call(params["k1"], prior_type, aux)
            k2 = self.k2.call(params["k2"], prior_type, aux)
            if prior_type is None:
                cov1 = k1(X1, X2)
                cov2 = k2(X1, X2)
                return self.operation(cov1, cov2)
            else:
                cov1, prior1 = k1(X1, X2)
                cov2, prior2 = k2(X1, X2)
                return self.operation(cov1, cov2), prior1 + prior2

        return kernel_fn

    def __initialize_params__(self, aux):
        params = {
            "k1": self.k1.__initialize_params__(aux),
            "k2": self.k2.__initialize_params__(aux),
        }
        self.constraints = {"k1": self.k1.constraints, "k2": self.k2.constraints}
        self.priors = {"k1": self.k1.priors, "k2": self.k2.priors}

        return params

    def __repr__(self) -> str:
        return f"({self.k1} {self.symbol} {self.k2})"


class Product(MathOperation):
    def __init__(self, k1, k2):
        super().__init__(k1, k2)
        self.operation = lambda k1, k2: k1 * k2
        self.symbol = "x"


class Sum(MathOperation):
    def __init__(self, k1, k2):
        super().__init__(k1, k2)
        self.operation = lambda k1, k2: k1 + k2
        self.symbol = "+"
