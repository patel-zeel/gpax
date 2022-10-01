from ._version import version as __version__  # noqa

from gpax.kernels import (
    RBFKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    PolynomialKernel,
)
from gpax.gps import ExactGP, SparseGP
from gpax.noises import HomoscedasticNoise
from gpax.means import ScalarMean
from gpax.special_kernels import GibbsKernel
from gpax.special_noises import HeteroscedasticNoise

import jax.numpy as jnp

jnp.jitter = jnp.array(1e-6)
print("imported nosthenogpax version")
