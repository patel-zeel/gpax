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


def randomize(params, priors, bijectors, key):
    seeds = seeds_like(params, key)

    def _randomize(param, prior, bijector, seed):
        sample = prior.sample(seed=seed, sample_shape=param.shape)
        if prior.__class__.__name__ == "Zero":
            return bijector(sample)
        else:
            return sample

    return tree_util.tree_map(
        lambda param, prior, bijector, seed: _randomize(param, prior, bijector, seed), params, priors, bijectors, seeds
    )


def seeds_like(params, key):
    values, treedef = tree_util.tree_flatten(params)
    keys = [key for key in jax.random.split(key, len(values))]
    return tree_util.tree_unflatten(treedef, keys)


def constrain(params, bijectors):
    return tree_util.tree_map(lambda param, bijector: bijector(param), params, bijectors)


def unconstrain(params, bijectors):
    return tree_util.tree_map(lambda param, bijector: bijector.inverse(param), params, bijectors)


def get_raw_log_prior(prior, params, bijectors):
    return tree_util.tree_map(
        lambda _prior, param, bijector: _prior.log_prob(param) - bijector.inverse_log_jacobian(param),
        prior,
        params,
        bijectors,
    )


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
