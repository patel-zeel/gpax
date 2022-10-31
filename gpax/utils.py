import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

distance_jitter = 0.0


def add_noise(K, noise, jitter):
    rows, columns = jnp.diag_indices_from(K)
    return K.at[rows, columns].set(K[rows, columns] + noise + jitter)


def vectorized_fn(fn, value, shape):
    for _ in shape:
        fn = jax.vmap(fn)
    return fn(value)


def squared_distance(X1, X2):
    return jnp.square(X1 - X2).sum() + distance_jitter


def distance(X1, X2):
    return jnp.sqrt(squared_distance(X1, X2))


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


def constrain(params, bijectors):
    return jtu.tree_map(lambda param, bijector: bijector(param), params, bijectors)


def unconstrain(params, bijectors):
    return jtu.tree_map(lambda param, bijector: bijector.inverse(param), params, bijectors)
