import jax
import jax.numpy as jnp
import lab.jax as B
from matrix import Dense
from plum import dispatch
from gpax.gps import ExactGP, RBFKernel

from mlkernels import Kernel, pairwise, elwise


class GibbsKernel(Kernel):
    def __init__(self, X_inducing, flex_scale=True, flex_variance=True, params=None):
        self.X_inducing = X_inducing
        self.flex_scale = flex_scale
        self.flex_variance = flex_variance
        self.params = params

    @staticmethod
    def predict_scale_per_dim(x, X_inducing, scale_gp_params, inducing_std_scale):
        X_inducing = B.uprank(X_inducing)
        scale_gp = ExactGP()
        covar = B.dense(scale_gp.kernel(scale_gp_params)(X_inducing, X_inducing))
        covar += jnp.eye(covar.shape[0]) * scale_gp_params["noise"]["variance"]
        latent_log_scale = scale_gp_params["mean"]["value"] + jnp.linalg.cholesky(covar) @ inducing_std_scale
        return B.exp(scale_gp.predict(scale_gp_params, X_inducing, latent_log_scale, x, return_cov=False)).squeeze()

    def predict_scale(self, x):
        f = jax.vmap(self.predict_scale_per_dim, in_axes=(None, 1, 0, 1))
        # print(jax.tree_util.tree_map(lambda x: x.shape, self.params["scale_gp"]))
        return f(x, self.X_inducing, self.params["scale_gp"], self.params["inducing_std_scale"])

    def predict_var(self, x):
        variance_gp = ExactGP(kernel=RBFKernel(active_dims=list(range(x.shape[1]))))
        covar = B.dense(variance_gp.kernel(self.params["variance_gp"])(self.X_inducing, self.X_inducing))
        covar += jnp.eye(covar.shape[0]) * self.params["variance_gp"]["noise"]["variance"]
        latent_log_variance = (
            self.params["variance_gp"]["mean"]["value"]
            + jnp.linalg.cholesky(covar) @ self.params["inducing_std_variance"]
        )
        return B.exp(
            variance_gp.predict(
                self.params["variance_gp"],
                self.X_inducing,
                latent_log_variance,
                x,
                return_cov=False,
            )
        ).squeeze()

    def _compute_pair(self, x1, x2):
        if self.flex_scale:
            predict_fn = jax.jit(self.predict_scale)
            l1 = predict_fn(x1.reshape(1, -1))
            l2 = predict_fn(x2.reshape(1, -1))
            l_avg_square = (l1**2 + l2**2) / 2.0
            l_avg = B.sqrt(l_avg_square)
            prefix_part = B.sqrt(l1 * l2 / l_avg_square).prod()
            x1_scaled = x1 / l_avg
            x2_scaled = x2 / l_avg
        else:
            lengthscale = self.params["lengthscale"]
            x1_scaled = x1 / lengthscale
            x2_scaled = x2 / lengthscale
            prefix_part = 1.0

        exp_part = B.exp(-0.5 * ((x1_scaled - x2_scaled) ** 2).sum())

        if self.flex_variance:
            predict_fn = jax.jit(self.predict_var)
            var1 = predict_fn(x1.reshape(1, -1))
            var2 = predict_fn(x2.reshape(1, -1))
            variance_part = var1 * var2
        else:
            variance_part = self.params["variance"]
        value = (variance_part * prefix_part * exp_part).squeeze()
        return value

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
