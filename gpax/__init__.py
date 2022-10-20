from ._version import version as __version__  # noqa

import warnings
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
from gpax.special_kernels import GibbsKernel, HeinonenGibbsKernel
from gpax.special_noises import HeteroscedasticNoise, HeinonenHeteroscedasticNoise

import jax
import jax.numpy as jnp

jnp.jitter = jnp.array(1e-6)

warnings.warn(f"jnp.jitter={jnp.jitter}")

jax.config.update("jax_enable_x64", True)
warnings.warn(f"64-bit precision={jax.config.read('jax_enable_x64')}")
