import jax
from jax.flatten_util import ravel_pytree
import jax.tree_util as tree_util
import optax


def randomize(params, key):
    values, unravel_fn = ravel_pytree(params)
    values = jax.random.normal(key, values.shape)
    return unravel_fn(values)


def constrain(params, bijectors):
    return tree_util.tree_map(lambda param, bijector: bijector(param), params, bijectors)


def unconstrain(params, bijectors):
    return tree_util.tree_map(lambda param, bijector: bijector.inverse(param), params, bijectors)


def train_fn(loss_fn, params, bijectors, optimizer, num_epochs=1):
    state = optimizer.init(params)

    constrained_loss_fn = lambda params: loss_fn(constrain(params, bijectors))

    # TODO: jitting this function does not work as of now
    def step(params_and_state, aux):
        params, state = params_and_state
        loss, grads = jax.value_and_grad(constrained_loss_fn)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), (params, loss)

    (params, state), (params_history, loss_history) = jax.lax.scan(
        f=step, init=(params, state), xs=None, length=num_epochs
    )
    return {
        "params": params,
        "params_history": params_history,
        "loss_history": loss_history,
    }
