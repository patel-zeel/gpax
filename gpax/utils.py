import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax.tree_util as tree_util
import optax
from gpax.distributions import NoPrior, TransformedDistribution

distance_jitter = 0.0


def squared_distance(X1, X2):
    return jnp.square(X1 - X2).sum() + distance_jitter


def distance(X1, X2):
    return jnp.sqrt(squared_distance(X1, X2) + distance_jitter)


def randomize(params, priors, bijectors, key, generic_sampler=jax.random.normal):
    seeds = seeds_like(params, key)

    def _randomize(param, prior, bijector, seed):
        if isinstance(prior, NoPrior):
            sample = generic_sampler(seed, param.shape)
            return bijector(sample)
        else:
            return prior.sample(seed=seed, sample_shape=param.shape)

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


def is_no_prior(prior):
    if isinstance(prior, TransformedDistribution):
        return is_no_prior(prior.distribution)
    else:
        return isinstance(prior, NoPrior)


def get_raw_log_prior(priors, params, bijectors):
    def _get_raw_log_prior(prior, param, bijector):
        if is_no_prior(prior):
            return prior.log_prob(param)
        else:
            return prior.log_prob(param) - bijector.inverse_log_jacobian(param)

    return tree_util.tree_map(
        lambda prior, param, bijector: _get_raw_log_prior(prior, param, bijector), priors, params, bijectors
    )


def train_fn(loss_fn, init_raw_params, optimizer, num_epochs=1, lax_scan=True):
    state = optimizer.init(init_raw_params)

    # dry run
    # loss_fn(init_raw_params)

    if lax_scan:
        value_and_grad_fn = jax.value_and_grad(loss_fn)

        def step(raw_params_and_state, aux):
            raw_params, state = raw_params_and_state
            loss, grads = value_and_grad_fn(raw_params)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            return (raw_params, state), (raw_params, loss)

        (raw_params, state), (raw_params_history, loss_history) = jax.lax.scan(
            f=step, init=(init_raw_params, state), xs=None, length=num_epochs
        )
    else:
        raw_params_history = []
        loss_history = []
        raw_params = init_raw_params
        grad_fn = jax.grad(loss_fn)
        for _ in range(num_epochs):
            loss = loss_fn(raw_params)
            grads = grad_fn(raw_params)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            raw_params_history.append(raw_params)
            loss_history.append(loss)
        loss_history = jnp.array(loss_history)
        raw_params_history = jnp.array(raw_params_history)
    return {
        "raw_params": raw_params,
        "raw_params_history": raw_params_history,
        "loss_history": loss_history,
    }
