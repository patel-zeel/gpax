import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from gpax.models import ExactGPRegression
import gpax.kernels as gk
import gpax.likelihoods as gl
import gpax.means as gm
import gpax.bijectors as gb
import gpax.distributions as gd
from gpax.plotting import plot_posterior

import optax
import regdata as rd

n_inducing = 10
X, y, X_test = rd.MotorcycleHelmet().get_data()
inducing_key = jax.random.PRNGKey(0)
X_inducing = jax.random.choice(inducing_key, X, (n_inducing,), replace=False)
optimizer_key = jax.random.PRNGKey(2)
optimizer = optax.adam(1e-2)
n_iters = 200

### Check
# l_prior = gb.get_positive_bijector()(gd.Normal(loc=0.0, scale=1.0))
# gp = ExactGPRegression(
#     kernel=gk.RBF(input_dim=X.shape[1], lengthscale_prior=l_prior), likelihood=gl.Gaussian(), mean=gm.Scalar()
# )
# key = jax.random.PRNGKey(1)
# gp.initialize(key)


latent_gp = ExactGPRegression(kernel=gk.RBF(input_dim=X.shape[1]), likelihood=gl.Gaussian(), mean=gm.Scalar())
likelihood = gl.HeteroscedasticGaussian(latent_gp=latent_gp)
gp = ExactGPRegression(
    kernel=gk.RBF(input_dim=X.shape[1]), likelihood=likelihood, mean=gm.Scalar(), X_inducing=X_inducing
)

### Test
# init_key = jax.random.PRNGKey(0)
# gp.initialize(init_key)
log_prob = gp.log_probability(X, y)
gp.unconstrain()
log_prior = gp.log_prior()

gp.constrain()
result = gp.optimize(optimizer_key, optimizer, X, y, n_iters=n_iters, lax_scan=True)

plt.plot(result["loss_history"])
plt.savefig("loss.png")

pred_mean, pred_cov = gp.predict(X, y, X_test)

fig, ax = plt.subplots()
plot_posterior(X, y, X_test, pred_mean, pred_cov, ax=ax)
fig.savefig("test.png")
