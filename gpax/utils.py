import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import optax

distance_jitter = 1e-36


def index_pytree(pytree, index):
    return jtu.tree_map(lambda x: x[index], pytree)


class DataScaler:
    def __init__(self, X, y=None, active_dims=None):
        if active_dims is not None:
            self.active_dims = active_dims
            all_dims = list(range(X.shape[1]))
            self.inactive_dims = sorted(set(all_dims) - set(self.active_dims))
            X_active = X[:, self.active_dims]
        else:
            self.active_dims = list(range(X.shape[1]))
            self.inactive_dims = []
            X_active = X

        self.X_min = X_active.min(axis=0)
        self.X_scale = X_active.max(axis=0) - self.X_min
        if y is not None:
            self.y_mean = y.mean()
            self.y_scale = jnp.max(jnp.abs(y - self.y_mean))

    def transform(self, X=None, y=None, ell=None, sigma=None, omega=None):
        res = []
        if X is not None:

            def transform_(x):
                x_new = (x[:, self.active_dims] - self.X_min) / self.X_scale
                return x.at[:, self.active_dims].set(x_new)

            X = jtu.tree_map(transform_, X)
            res.append(X)
        if y is not None:
            fn = lambda x: (x - self.y_mean) / self.y_scale
            res.append(jtu.tree_map(fn, y))
        if ell is not None:
            fn = lambda x: x / self.X_scale
            res.append(jtu.tree_map(fn, ell))
        if sigma is not None:
            fn = lambda x: x / self.y_scale
            res.append(jtu.tree_map(fn, sigma))
        if omega is not None:
            fn = lambda x: x / self.y_scale
            res.append(jtu.tree_map(fn, omega))
        return res

    def inverse_transform(self, X=None, y=None, ell=None, sigma=None, omega=None):
        res = []
        if X is not None:

            def inv_trans_fn(x):
                x_new = x[:, self.active_dims] * self.X_scale + self.X_min
                return x.at[:, self.active_dims].set(x_new)

            X = jtu.tree_map(inv_trans_fn, X)
            res.append(X)
        if y is not None:
            fn = lambda x: x * self.y_scale + self.y_mean
            res.append(jtu.tree_map(fn, y))
        if ell is not None:
            fn = lambda x: x * self.X_scale
            res.append(jtu.tree_map(fn, ell))
        if sigma is not None:
            fn = lambda x: x * self.y_scale
            res.append(jtu.tree_map(fn, sigma))
        if omega is not None:
            fn = lambda x: x * self.y_scale
            res.append(jtu.tree_map(fn, omega))
        return res


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


def squared_distance(x1, x2):
    return jnp.square(x1 - x2).sum()


def distance(x1, x2):
    return jnp.sqrt(squared_distance(x1, x2) + distance_jitter)
    # return jnp.sqrt(jnp.max(squared_distance(x1, x2), distance_jitter))


def train_fn(loss_fn, init_raw_params, optimizer, n_iters=1, lax_scan=True):
    state = optimizer.init(init_raw_params)

    # dry run
    # loss_fn(init_raw_params)

    if lax_scan:
        value_and_grad_fn = jax.value_and_grad(loss_fn)

        @jax.jit
        def step(raw_params_and_state, aux):
            raw_params, state = raw_params_and_state
            loss, grads = value_and_grad_fn(raw_params)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            return (raw_params, state), (raw_params, loss)

        (raw_params, state), (raw_params_history, loss_history) = jax.lax.scan(
            f=step, init=(init_raw_params, state), xs=None, length=n_iters
        )
    else:
        raw_params_history = []
        loss_history = []
        raw_params = init_raw_params
        grad_fn = jax.grad(loss_fn)
        for _ in range(n_iters):
            loss = loss_fn(raw_params)
            grads = grad_fn(raw_params)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            raw_params_history.append(raw_params)
            loss_history.append(loss)
        loss_history = jnp.array(loss_history)
    return {
        "raw_params": raw_params,
        "raw_params_history": raw_params_history,
        "loss_history": loss_history,
    }


def constrain(params, bijectors):
    return jtu.tree_map(lambda param, bijector: bijector(param), params, bijectors)


def unconstrain(params, bijectors):
    return jtu.tree_map(lambda param, bijector: bijector.inverse(param), params, bijectors)
