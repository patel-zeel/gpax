import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax.tree_util as tree_util
import optax
from gpax.distributions import Zero


def squared_distance(X1, X2):
    return jnp.square(X1 - X2).sum()


def distance(X1, X2):
    return jnp.sqrt(squared_distance(X1, X2))


def randomize(params, key):
    values, unravel_fn = ravel_pytree(params)
    values = jax.random.normal(key, values.shape)
    return unravel_fn(values)


def constrain(params, bijectors):
    return tree_util.tree_map(lambda param, bijector: bijector(param), params, bijectors)


def unconstrain(params, bijectors):
    return tree_util.tree_map(lambda param, bijector: bijector.inverse(param), params, bijectors)


def initialize_zero_prior(params):
    tree_util.tree_map(lambda param: Zero(), params)


def train_fn(loss_fn, init_raw_params, optimizer, num_epochs=1):
    state = optimizer.init(init_raw_params)

    # dry run
    loss_fn(init_raw_params)

    @jax.jit
    def step(raw_params_and_state, aux):
        raw_params, state = raw_params_and_state
        loss, grads = jax.value_and_grad(loss_fn)(raw_params)
        updates, state = optimizer.update(grads, state)
        raw_params = optax.apply_updates(raw_params, updates)
        return (raw_params, state), (raw_params, loss)

    (raw_params, state), (raw_params_history, loss_history) = jax.lax.scan(
        f=step, init=(init_raw_params, state), xs=None, length=num_epochs
    )
    return {
        "raw_params": raw_params,
        "raw_params_history": raw_params_history,
        "loss_history": loss_history,
    }
