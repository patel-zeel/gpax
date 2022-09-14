import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_posterior(model, params, X, y, X_test, ax, include_noise=True, alpha=0.3):
    if X.shape[1] > 1:
        raise NotImplementedError("Only 1D inputs are supported")

    if ax is None:
        ax = plt.gca()

    pred_mean, pred_cov = model.predict(params, X, y, X_test, include_noise=include_noise)
    pred_mean = pred_mean.squeeze()
    pred_std = jnp.sqrt(jnp.diag(pred_cov))
    ax.scatter(X, y, label="Observations")
    ax.plot(X_test, pred_mean, label="Posterior mean")
    ax.fill_between(
        X_test.ravel(),
        pred_mean - 2 * pred_std,
        pred_mean + 2 * pred_std,
        alpha=alpha,
        label="95% confidence interval",
    )
    return ax
