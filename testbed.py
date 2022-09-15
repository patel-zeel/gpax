import jax
import jax.numpy as jnp

from gpax import GibbsKernel
from gpax.utils import randomize, constrain, unconstrain
from gpflex.core.kernels import FlexKernel
from tinygp import GaussianProcess, kernels, transforms


X = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
y = jax.random.normal(jax.random.PRNGKey(1), (5,))

X_inducing = X[:2]
print(X.shape, X_inducing.shape)

gkernel = GibbsKernel(X_inducing=X_inducing)
params = gkernel.initialise_params(key=jax.random.PRNGKey(0), X_inducing=X_inducing)
bijectors = gkernel.get_bijectors()
params = unconstrain(params, bijectors)
# params = randomize(params, key=jax.random.PRNGKey(1))
cons_params = constrain(params, bijectors)


gp = GaussianProcess(
    kernel=cons_params["kernel"]["variance_gp"]["kernel"]["variance"].squeeze()
    * transforms.Linear(
        1 / cons_params["kernel"]["variance_gp"]["kernel"]["lengthscale"].squeeze(), kernels.ExpSquared(scale=1.0)
    ),
    X=cons_params["kernel"]["X_inducing"],
    diag=cons_params["kernel"]["variance_gp"]["noise"]["variance"].squeeze(),
)
new = gp.condition(cons_params["kernel"]["latent_log_variance"], X).gp
print(new.mean)

print(gkernel(cons_params)(X, X))

fkernel = FlexKernel(latent_x=X_inducing)
print(fkernel(X, X))
pass
