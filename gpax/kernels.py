from jax import vmap
import jax.numpy as jnp
from typing import Callable, Dict, List, Optional, Tuple, Union
from jaxtyping import Array, Float
from gpjax.kernels import AbstractKernel, AbstractKernelComputation
from jaxlinop import LinearOperator
from chex import PRNGKey as PRNGKeyType


class GibbsKernelComputation(AbstractKernelComputation):
    def __init__(
        self,
        kernel_fn: Callable[[Dict, Float[Array, "N D"], Float[Array, "M D"]], Array] = None,
        flex_ell: Optional[bool] = True,
        flex_sigma: Optional[bool] = True,
        flex_omega: Optional[bool] = True,
    ) -> None:
        super().__init__(kernel_fn)
        self.flex_ell = flex_ell
        self.flex_sigma = flex_sigma
        self.flex_omega = flex_omega

    def cross_covariance(self, params: Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]) -> Float[Array, "N M"]:
        """For a given kernel, compute the NxM covariance matrix on a pair of input
        matrices of shape NxD and MxD.

        Args:
            kernel (AbstractKernel): The kernel for which the Gram
                matrix should be computed for.
            params (Dict): The kernel's parameter set.
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.

        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """
        cross_cov = vmap(lambda x: vmap(lambda y: self.kernel_fn(params, x, y))(y))(x)
        return cross_cov

    def gram(
        self,
        params: Dict,
        inputs: Float[Array, "N D"],
    ) -> Tuple[LinearOperator, float]:
        pass


# Implement Heinonen first then inherit from it making a master class
class GibbsKernel(AbstractKernel):
    def __init__(
        self,
        compute_engine: AbstractKernelComputation = None,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "GibbsKernel",
        ARD: Optional[bool] = True,
        flex_ell: Optional[bool] = True,
        flex_sigma: Optional[bool] = True,
        flex_omega: Optional[bool] = True,
    ) -> None:
        compute_engine = lambda kernel_fn: GibbsKernelComputation(kernel_fn, flex_ell, flex_sigma, flex_omega)
        super().__init__(compute_engine, active_dims, stationary, spectral, name)
        self.ARD = ARD

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
        flex_ell: Optional[bool] = True,
        flex_sigma: Optional[bool] = True,
        flex_omega: Optional[bool] = True,
        compute_log_prior: Optional[bool] = False,
    ) -> Union[Tuple[Float[Array, "N, M"], float], Tuple[Float[Array, "N, M"]]]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( \\frac{\\lVert x - y \\rVert^2_2}{2 \\ell^2} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x)
        y = self.slice_input(y)

        # --------------------
        # K = params["variance"] * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        if self.flex_ell:
            ell = jnp.ones(self.input_dim)
