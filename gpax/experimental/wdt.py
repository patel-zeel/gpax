import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

m = 1.0
s = 2.0

def loss_fn(w):
    p = m + s * w
    return t