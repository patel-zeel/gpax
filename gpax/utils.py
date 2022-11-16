import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import optax

distance_jitter = 0.0


def repeat_to_size(value, size):
    if value.size == 1:
        return jnp.repeat(value, size)
    elif value.size == size:
        return value
    else:
        raise ValueError("value.size must be 1 or size")


def add_to_diagonal(K, value, jitter):
    diag_indices = jnp.diag_indices_from(K)
    return K.at[diag_indices].set(K[diag_indices] + value + jitter)


def get_a_inv_b(a, b, return_cholesky=False):
    chol = jnp.linalg.cholesky(a)
    k_inv_y = jsp.linalg.cho_solve((chol, True), b)
    if return_cholesky:
        return k_inv_y, chol
    else:
        return k_inv_y


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
