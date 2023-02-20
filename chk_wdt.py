import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
from gpax.kernels import Hamming


x = jnp.arange(4).reshape(-1, 1)

K = Hamming(x).get_kernel_fn()(x, x)
print(K)
