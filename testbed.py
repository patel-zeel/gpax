from time import time

init = time()
import distrax

midtime = time()
import tensorflow_probability.substrates.jax as tfp


print(f"TFP: {midtime - init} seconds")
print(f"Distrax: {time() - midtime} seconds")
