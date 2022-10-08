import jax

import lab.jax as B
from matrix import Dense
from plum import dispatch
from gpax.gps import ExactGP

from mlkernels import Kernel, pairwise, elwise


class GibbsKernel(Kernel):
    def __init__(self, X_inducing, flex_scale=True, flex_variance=True, params=None):
        self.X_inducing = X_inducing
        self.flex_scale = flex_scale
        self.flex_variance = flex_variance
        self.params = params

    @staticmethod
    def predict_scale_per_dim(x, X_inducing, scale_gp_params, latent_log_scale):
        scale_gp = ExactGP()
        return B.exp(
            scale_gp.predict(
                scale_gp_params, X_inducing.reshape(-1, 1), latent_log_scale, x.reshape(-1, 1), return_cov=False
            )
        ).squeeze()

    def predict_scale(self, x):
        f = jax.vmap(self.predict_scale_per_dim, in_axes=(None, 1, 0, 1))
        return f(x, self.X_inducing, self.params["scale_gp"], self.params["latent_log_scale"])

    def predict_variance(self, x):
        variance_gp = ExactGP()
        return B.exp(
            variance_gp.predict(
                self.params["variance_gp"],
                self.X_inducing,
                self.params["latent_log_variance"],
                x,
                return_cov=False,
            )
        ).squeeze()

    def _compute_pair(self, x1, x2):
        if self.flex_scale:
            predict_fn = jax.jit(self.predict_scale)
            l1 = predict_fn(x1)
            l2 = predict_fn(x2)
            l_avg_square = (l1**2 + l2**2) / 2.0
            l_avg = B.sqrt(l_avg_square)
            prefix_part = B.sqrt(l1 * l2 / l_avg_square).prod()
            x1 = x1 / l_avg
            x2 = x2 / l_avg
        else:
            lengthscale = self.params["lengthscale"]
            x1 = x1 / lengthscale
            x2 = x2 / lengthscale
            prefix_part = 1.0

        exp_part = B.exp(-0.5 * ((x1 - x2) ** 2).sum())

        if self.flex_variance:
            predict_fn = jax.jit(self.predict_variance)
            var1 = predict_fn(x1)
            var2 = predict_fn(x2)
            variance_part = var1 * var2
        else:
            variance_part = self.params["variance"]

        return (variance_part * prefix_part * exp_part).squeeze()

    def _compute_pairwise(self, x1, x2):
        f = jax.vmap(self._compute_pair, in_axes=(None, 0))
        f = jax.vmap(f, in_axes=(0, None))
        return f(x1, x2)

    def _compute_elwise(self, x1, x2):
        f = jax.vmap(self._compute_pair, in_axes=(0, 0))
        return f(x1, x2)


@pairwise.dispatch
def pairwise(k: GibbsKernel, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute_pairwise(x, y))


@elwise.dispatch
def elwise(k: GibbsKernel, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute_elwise(x, y))
