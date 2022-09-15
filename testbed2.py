import jax
from tinygp import GaussianProcess, kernels, transforms
from gpax import GibbsKernel, ExactGP, HomoscedasticNoise

x = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
x_test = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
y = jax.random.normal(jax.random.PRNGKey(2), (5,))

gp = GaussianProcess(kernel=0.5 * kernels.ExpSquared(scale=0.8), X=x, diag=0.1)
pred_gp = gp.condition(y, x_test).gp
print(pred_gp.mean)

gp = ExactGP(
    kernel=GibbsKernel(flex_scale=False, flex_variance=False, active_dims=[0, 1]),
    noise=HomoscedasticNoise(variance=0.1),
)
params = gp.initialise_params(key=jax.random.PRNGKey(0), X=x)
params["kernel"]["lengthscale"] = 0.8
params["kernel"]["variance"] = 0.5
print(gp.predict(params, x, y, x_test, return_cov=False))
pass
