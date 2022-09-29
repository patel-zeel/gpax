import jax
import jax.numpy as jnp
import jax.scipy as jsp

from gpax.distributions import TransformedDistribution, Distribution


@jax.tree_util.register_pytree_node_class
class Bijector:
    def __call__(self, ele):
        if isinstance(ele, Distribution):
            return TransformedDistribution(ele, self)
        else:
            return self.forward_fn(ele)

    def forward(self, ele):
        return self(ele)

    def inverse(self, ele):
        if isinstance(ele, Distribution):
            return TransformedDistribution(ele, invert_bijector(self))
        else:
            return self.inverse_fn(ele)

    def log_jacobian(self, value):
        return jnp.log(jax.jacobian(self.forward)(value))

    def inverse_log_jacobian(self, array):
        def _inverse_log_jacobian(value):
            return jnp.log(jax.jacobian(self.inverse)(value))

        for _ in array.shape:
            _inverse_log_jacobian = jax.vmap(_inverse_log_jacobian)
        return _inverse_log_jacobian(array)

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


@jax.tree_util.register_pytree_node_class
class Log(Bijector):
    def __init__(self):
        self.forward_fn = jnp.log
        self.inverse_fn = jnp.exp


@jax.tree_util.register_pytree_node_class
class Exp(Bijector):
    def __init__(self):
        self.forward_fn = jnp.exp
        self.inverse_fn = jnp.log


@jax.tree_util.register_pytree_node_class
class Sigmoid(Bijector):
    def __init__(self):
        self.forward_fn = jax.nn.sigmoid
        self.inverse_fn = jsp.special.logit


@jax.tree_util.register_pytree_node_class
class Identity(Bijector):
    def __init__(self):
        self.forward_fn = lambda x: x
        self.inverse_fn = lambda x: x


def invert_bijector(bijector):
    bijector.forward_fn, bijector.inverse_fn = bijector.inverse_fn, bijector.forward_fn
    return bijector
