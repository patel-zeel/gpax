import jax
import jax.numpy as jnp
import jax.scipy as jsp


def rmse(y_true, y_pred):
    """Root mean squared error."""
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 1
    assert y_pred.ndim == 1
    return jnp.sqrt(jnp.mean((y_true - y_pred) ** 2))


def msll(y_true, y_pred_mean, y_pred_var):
    """Mean Standardized Log Likelihood."""
    assert y_true.shape == y_pred_mean.shape == y_pred_var.shape
    assert y_true.ndim == 1
    assert y_pred_mean.ndim == 1
    assert y_pred_var.ndim == 1
