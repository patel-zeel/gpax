import jax
from gpax.kernels import Kernel, RBFKernel
from gpax.noises import Noise, HomoscedasticNoise
from gpax.means import Mean, ConstantMean

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from stheno.jax import GP as _GPStheno, PseudoObs, PseudoObsFITC, PseudoObsDTC
import lab.jax as B
from jaxtyping import Array
from chex import dataclass


@dataclass
class AbstractGP:
    kernel: Kernel = RBFKernel()
    noise: Noise = HomoscedasticNoise()
    mean: Mean = ConstantMean()

    def get_gp(self, params):
        mean = self.mean(params)
        kernel = self.kernel(params)
        return _GPStheno(mean, kernel)

    def return_posterior(self, params, f, obs, X_test, return_cov=True, include_noise=True):
        posterior = f | obs
        if include_noise:
            post_pred = posterior(X_test, self.noise(params, X_test))
        else:
            post_pred = posterior(X_test)
        post_mean = B.dense(post_pred.mean)
        if return_cov:
            post_cov = B.dense(post_pred.var)
            return post_mean, post_cov
        else:
            return post_mean

    def initialise_params(self, key, X, X_inducing=None):
        keys = jax.random.split(key, 4)
        if self.kernel.__class__.__name__ == "GibbsKernel":
            kernels_params = self.kernel.initialise_params(keys[0], X_inducing=X_inducing)
        elif self.kernel.__class__.__name__ in ["SumKernel", "ProductKernel"]:
            kernels_params = self.kernel.initialise_params(keys[0], X=X, X_inducing=X_inducing)
        else:
            kernels_params = self.kernel.initialise_params(keys[0], X=X)
        params = {
            **self.mean.initialise_params(keys[0]),
            **kernels_params,
            **self.noise.initialise_params(keys[2], X_inducing=X_inducing),
        }
        key = jax.random.split(keys[3], 1)[0]
        return {**params, **self.__initialise_params__(key, X=X, X_inducing=X_inducing)}

    def __initialise_params__(self, key, X, X_inducing=None):
        return NotImplementedError("This method must be implemented by a subclass.")

    def get_bijectors(self):
        bijectors = {
            **self.mean.get_bijectors(),
            **self.kernel.get_bijectors(),
            **self.noise.get_bijectors(),
        }
        return {**bijectors, **self.__get_bijectors__()}

    def __get_bijectors__(self):
        return NotImplementedError("This method must be implemented by a subclass.")


@dataclass
class ExactGP(AbstractGP):
    def log_probability(self, params, X, y):
        gp = self.get_gp(params)
        noise = self.noise(params, X)
        return gp(X, noise).logpdf(y)

    def predict(self, params, X, y, X_test, return_cov=True, include_noise=True):
        gp = self.get_gp(params)
        train_noise = self.noise(params, X)
        obs = (gp(X, train_noise), y)
        return self.return_posterior(params, gp, obs, X_test, return_cov, include_noise)

    def __initialise_params__(self, key, X, X_inducing):
        return {}  # No additional parameters to initialise.

    def __get_bijectors__(self):
        return {}  # No additional bijectors to return.


@dataclass
class SparseGP(AbstractGP):
    method: str = "vfe"

    def __post_init__(self):
        assert self.method in [
            "vfe",
            "fitc",
            "dtc",
        ], "method must be one of vfe, fitc, dtc"

        self.pseudo_obs = {"vfe": PseudoObs, "fitc": PseudoObsFITC, "dtc": PseudoObsDTC}

    def get_pseudo_obs(self, params, f, X, y):
        train_noise = self.noise(params, X)
        return self.pseudo_obs[self.method](f(params["X_inducing"]), f(X, train_noise), y)

    def elbo(self, params, X, y):
        f = self.get_gp(params)
        pseudo_obs = self.get_pseudo_obs(params, f, X, y)
        elbo = pseudo_obs.elbo(f.measure)
        return elbo

    def predict(self, params, X, y, X_test, return_cov=True, include_noise=True):
        f = self.get_gp(params)
        pseudo_obs = self.get_pseudo_obs(params, f, X, y)
        return self.return_posterior(params, f, pseudo_obs, X_test, return_cov, include_noise)

    def __initialise_params__(self, key, X, X_inducing):
        assert X_inducing is not None, "X_inducing must be provided for sparse GPs."
        return {"X_inducing": X_inducing}

    def __get_bijectors__(self):
        return {"X_inducing": tfb.Identity()}
