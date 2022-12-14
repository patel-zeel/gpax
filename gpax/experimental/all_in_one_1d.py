from time import time

import jaxopt

checkpoint = time()


def time_it(label):
    global checkpoint
    new = time()
    print(f"{label}: Time: {new - checkpoint:.2f} seconds")
    checkpoint = new


import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from functools import partial

import optax

from gpax.kernels import RBF
import regdata as rd

import matplotlib.pyplot as plt

from gpax.plotting import plot_posterior
from gpax.utils import add_to_diagonal, squared_distance, get_a_inv_b, repeat_to_size, train_fn
import gpax.distributions as gd
import gpax.bijectors as gb
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

jax.config.update("jax_enable_x64", True)

time_it("Imports")

jitter = 1e-6
cs = jsp.linalg.cho_solve
st = jsp.linalg.solve_triangular

dist_f = jax.vmap(squared_distance, in_axes=(None, 0))
dist_f = jax.vmap(dist_f, in_axes=(0, None))


def get_simulated_data(flex_scale=False, flex_var=False, flex_noise=False):
    key = jax.random.PRNGKey(1221)  # was 1221
    n_points = 200
    fn_dict = {}

    def kernel_fn(x1, ls1, var1, x2, ls2, var2):
        l_sqr_avg = (ls1**2 + ls2**2) / 2
        prefix = jnp.sqrt(ls1 * ls2 / l_sqr_avg)
        exp_part = jnp.exp(-0.5 * ((x1 - x2) ** 2) / l_sqr_avg)
        return (var1 * var2 * prefix * exp_part).squeeze()

    def add_noise(K, noise):
        rows, columns = jnp.diag_indices_from(K)
        return K.at[rows, columns].set(K[rows, columns] + noise.ravel() + 10e-6)

    kernel_fn = jax.vmap(kernel_fn, in_axes=(None, None, None, 0, 0, 0))
    kernel_fn = jax.vmap(kernel_fn, in_axes=(0, 0, 0, None, None, None))

    keys = jax.random.split(key, 3)
    if flex_scale:
        scale_fn = lambda x: (0.5 * jnp.sin(x / 8)) + 1.5
    else:
        scale_fn = lambda x: jnp.array(1.0).repeat(x.size).reshape(x.shape)

    if flex_var:
        var_fn = lambda x: 1.5 * jnp.exp(jnp.sin(0.2 * x))  # jnp.exp(jnp.sin(x))
    #         var_fn = lambda x:  jax.nn.softplus(gp.sample(keys[1])) #(1.1 + jnp.cos(x - jnp.pi / 2)) / 2
    else:
        var_fn = lambda x: jnp.array(1.0).repeat(x.size).reshape(x.shape)
    if flex_noise:
        noise_fn = lambda x: 2.5 * jax.nn.softplus(jnp.sin(0.2 * -x))  # jnp.exp(jnp.sin(-x))
    #         noise_fn = lambda x: jax.nn.softplus(gp.sample(keys[2]))/2 # (1.1 + jnp.sin(x + jnp.pi / 2)) / 4
    else:
        noise_fn = lambda x: jnp.array(0.1).repeat(x.size).reshape(x.shape)

    fn_dict["lengthscale"] = scale_fn
    fn_dict["scale"] = var_fn
    fn_dict["noise"] = noise_fn
    #     lengthscale_trend = lambda x: (0.5 * jnp.sin(5 * x / 8)) + 1.0
    #     variance_trend = lambda x: jnp.exp(jnp.sin(0.2 * x))  # (0.3 * x**2) + 0.4
    #     noise_var_trend = lambda x: jnp.exp(jnp.sin(0.2 * -x))

    #     n_points = 125
    x = jnp.linspace(-30, 30, n_points).reshape(-1, 1)
    #     print(x.shape, scale_fn(x).shape, var_fn(x).shape)
    # gp = GaussianProcess(kernel=1.0 * kernels.ExpSquared(scale=0.9), X=x)
    covar = kernel_fn(x, scale_fn(x), var_fn(x), x, scale_fn(x), var_fn(x))
    covar = add_noise(covar, jnp.array(0.0))

    true_fn = jnp.linalg.cholesky(covar) @ jax.random.normal(key, (n_points,))
    #     print(true_fn.shape, jax.random.normal(key, true_f.shape).shape)

    key = jax.random.split(key, 1)[0]
    y = true_fn + jax.random.normal(key, true_fn.shape).ravel() * (noise_fn(x).ravel() ** 0.5)

    return x, y, fn_dict, true_fn


def get_latent_chol(X, ell, sigma):
    kernel_fn = RBF(input_dim=X.shape[1], lengthscale=ell, scale=sigma)
    cov = kernel_fn(X, X)
    noisy_cov = add_to_diagonal(cov, 0.0, jitter)
    chol = jnp.linalg.cholesky(noisy_cov)
    return chol, kernel_fn


def get_white(h, X, ell, sigma):
    log_h = jnp.log(repeat_to_size(h, X.shape[0]))
    log_h_bar = log_h - jnp.mean(log_h)
    chol, _ = get_latent_chol(X, ell, sigma)
    return st(chol, log_h_bar, lower=True)


def get_log_h(white_h, h_mean, X, ell, sigma):
    chol, kernel_fn = get_latent_chol(X, ell, sigma)
    return h_mean + chol @ white_h, chol, kernel_fn


def predict_h(white_h, h_mean, X, X_new, ell, sigma):
    log_h, chol, kernel_fn = get_log_h(white_h, h_mean, X, ell, sigma)
    K_star = kernel_fn(X_new, X)
    # return jnp.exp(log_h), jnp.exp(log_h.mean() + K_star @ cs((chol, True), log_h - log_h.mean()))
    return jnp.exp(log_h), jnp.exp(h_mean + K_star @ cs((chol, True), log_h - h_mean))


def gibbs_k(X1, X2, ell1, ell2, s1, s2):
    ell1, ell2 = ell1.reshape(-1, 1), ell2.reshape(-1, 1)  # 1D only
    l_avg_square = (ell1**2 + ell2.T**2) / 2.0
    prefix_part = jnp.sqrt(ell1 * ell2.T / l_avg_square)
    squared_dist = dist_f(X1, X2)
    exp_part = jnp.exp(-squared_dist / (2.0 * l_avg_square))
    s1, s2 = s1.reshape(-1, 1), s2.reshape(-1, 1)  # 1D only
    variance = s1 * s2.T
    return variance * prefix_part * exp_part, locals()


def value_and_grad_fn(params, X, y):
    for name in ["ell", "sigma", "omega"]:
        params[f"{name}_gp_log_ell"] = jax.lax.stop_gradient(params[f"{name}_gp_log_ell"])
        params[f"{name}_gp_log_sigma"] = jax.lax.stop_gradient(params[f"{name}_gp_log_sigma"])

    grads = {}
    log_ell, chol_ell, _ = get_log_h(
        params["white_ell"],
        params["white_ell_mean"],
        X,
        ell=jnp.exp(params["ell_gp_log_ell"]),
        sigma=jnp.exp(params["ell_gp_log_sigma"]),
    )
    log_omega, chol_omega, _ = get_log_h(
        params["white_omega"],
        params["white_omega_mean"],
        X,
        ell=jnp.exp(params["omega_gp_log_ell"]),
        sigma=jnp.exp(params["omega_gp_log_sigma"]),
    )
    log_sigma, chol_sigma, _ = get_log_h(
        params["white_sigma"],
        params["white_sigma_mean"],
        X,
        ell=jnp.exp(params["sigma_gp_log_ell"]),
        sigma=jnp.exp(params["sigma_gp_log_sigma"]),
    )

    (ell, omega, sigma) = jnp.exp(log_ell), jnp.exp(log_omega), jnp.exp(log_sigma)

    K_f, aux = gibbs_k(X, X, ell, ell, sigma, sigma)
    K_y = add_to_diagonal(K_f, omega**2, 0.0)

    # ### Manual Grads
    # a, chol_y = get_a_inv_b(K_y, y, return_cholesky=True)
    # aat = a.reshape(-1, 1) @ a.reshape(1, -1)

    # ## omega
    # o2 = cs((chol_omega, True), log_omega - log_omega.mean())
    # o1 = aat @ jnp.diag(omega**2) - cs((chol_y, True), jnp.diag(omega**2))
    # grads["white_omega"] = chol_omega.T @ (jnp.diag(o1) - o2)

    # ## sigma
    # s2 = cs((chol_sigma, True), log_sigma - log_sigma.mean())
    # s1 = aat @ K_f - cs((chol_y, True), K_f)
    # grads["white_sigma"] = chol_sigma.T @ (jnp.diag(s1) - s2)

    # ## ell
    # dK = (
    #     (aux["variance"] / aux["prefix_part"] * aux["exp_part"] / aux["l_avg_square"] ** 3 / 8)
    #     * (ell.reshape(-1, 1) * ell.reshape(1, -1))
    #     * (4 * aux["squared_dist"] * ell.reshape(-1, 1) ** 2 - ell.reshape(-1, 1) ** 4 + ell.reshape(1, -1) ** 4)
    # )
    # l2 = cs((chol_ell, True), log_ell - log_ell.mean())
    # l1 = aat @ dK - cs((chol_y, True), dK)
    # grads["white_ell"] = chol_ell.T @ (jnp.diag(l1) - l2)

    mu_f = 0
    log_lik = tfd.MultivariateNormalFullCovariance(loc=mu_f, covariance_matrix=K_y).log_prob(y)

    # Type - A - Prior on correlated parameters
    # log_prior_ell = tfd.MultivariateNormalTriL(loc=params["white_ell_mean"], scale_tril=chol_ell).log_prob(log_ell)
    # log_prior_omega = tfd.MultivariateNormalTriL(loc=params["white_omega_mean"], scale_tril=chol_omega).log_prob(
    #     log_omega
    # )
    # log_prior_sigma = tfd.MultivariateNormalTriL(loc=params["white_sigma_mean"], scale_tril=chol_sigma).log_prob(
    #     log_sigma
    # )

    # Type -B - Prior on White parameters
    # log_prior_ell = (
    #     tfd.Normal(st(chol_ell, params["white_ell_mean"].repeat(X.shape[0]), lower=True), 1.0)
    #     .log_prob(params["white_ell"])
    #     .sum()
    # )
    # log_prior_omega = (
    #     tfd.Normal(st(chol_omega, params["white_omega_mean"].repeat(X.shape[0]), lower=True), 1.0)
    #     .log_prob(params["white_omega"])
    #     .sum()
    # )
    # log_prior_sigma = (
    #     tfd.Normal(st(chol_sigma, params["white_sigma_mean"].repeat(X.shape[0]), lower=True), 1.0)
    #     .log_prob(params["white_sigma"])
    #     .sum()
    # )

    # Type -C - Prior on Standard Normal parameters
    log_prior_ell = tfd.Normal(0.0, 1.0).log_prob(params["white_ell"]).sum()
    log_prior_omega = tfd.Normal(0.0, 1.0).log_prob(params["white_omega"]).sum()
    log_prior_sigma = tfd.Normal(0.0, 1.0).log_prob(params["white_sigma"]).sum()

    lgp_ell_prior = 0.0
    lgp_sigma_prior = 0.0
    # ll_prior_d = gd.Frechet(rate=-jnp.log(0.5) * (0.2**0.5), dim=1)
    lgp_ell_prior_d = gd.Normal(loc=-1.2, scale=0.5)
    lgp_sigma_prior_d = gd.Normal(loc=0.0, scale=0.5)

    lgp_ell_prior += lgp_ell_prior_d.log_prob(params["ell_gp_log_ell"]).sum()
    lgp_ell_prior += lgp_ell_prior_d.log_prob(params["omega_gp_log_ell"]).sum()
    lgp_ell_prior += lgp_ell_prior_d.log_prob(params["sigma_gp_log_ell"]).sum()

    lgp_sigma_prior += lgp_sigma_prior_d.log_prob(params["ell_gp_log_sigma"]).sum()
    lgp_sigma_prior += lgp_sigma_prior_d.log_prob(params["omega_gp_log_sigma"]).sum()
    lgp_sigma_prior += lgp_sigma_prior_d.log_prob(params["sigma_gp_log_sigma"]).sum()

    print(
        "log_lik",
        log_lik,
        "log_prior_ell",
        log_prior_ell,
        "log_prior_omega",
        log_prior_omega,
        "log_prior_sigma",
        log_prior_sigma,
        "lgp_ell_prior",
        lgp_ell_prior,
        "lgp_sigma_prior",
        lgp_sigma_prior,
    )

    return -(log_lik + log_prior_ell + log_prior_omega + log_prior_sigma + lgp_ell_prior + lgp_sigma_prior)  # , grads

    # Use this for debugging
    # auto_grad, manual_grad = jax.grad(value_and_grad_fn, has_aux=True)(params, X, y)


def predict_fn(params, X, y, X_new):
    ell, ell_new = predict_h(
        params["white_ell"],
        params["white_ell_mean"],
        X,
        X_new,
        ell=jnp.exp(params["ell_gp_log_ell"]),
        sigma=jnp.exp(params["ell_gp_log_sigma"]),
    )
    omega, omega_new = predict_h(
        params["white_omega"],
        params["white_omega_mean"],
        X,
        X_new,
        ell=jnp.exp(params["omega_gp_log_ell"]),
        sigma=jnp.exp(params["omega_gp_log_sigma"]),
    )
    sigma, sigma_new = predict_h(
        params["white_sigma"],
        params["white_sigma_mean"],
        X,
        X_new,
        ell=jnp.exp(params["sigma_gp_log_ell"]),
        sigma=jnp.exp(params["sigma_gp_log_sigma"]),
    )

    K, _ = gibbs_k(X, X, ell, ell, sigma, sigma)
    K_noisy = add_to_diagonal(K, omega**2, jitter)
    chol_y = jnp.linalg.cholesky(K_noisy)

    K_star, _ = gibbs_k(X_new, X, ell_new, ell, sigma_new, sigma)
    K_star_star, _ = gibbs_k(X_new, X_new, ell_new, ell_new, sigma_new, sigma_new)

    pred_mean = K_star @ cs((chol_y, True), y)
    pred_cov = K_star_star - K_star @ cs((chol_y, True), K_star.T)
    return pred_mean, pred_cov, omega_new**2


#### data
data = rd.MotorcycleHelmet
X, y, X_test = data().get_data()
data_name = data.__name__

# X, y, _, _ = get_simulated_data(flex_scale=1, flex_noise=1, flex_var=1)
# data_name = "simulated"
x_scaler = MinMaxScaler()
scale = 1
X = x_scaler.fit_transform(X) * scale
y = (y - y.mean()) / jnp.max(jnp.abs(y - jnp.mean(y)))

X_test = jnp.linspace(-2 * scale, 3 * scale, 250).reshape(-1, 1)
# X_test = jnp.linspace(-40, 40, 250).reshape(-1, 1)

time_it("Data loaded")

params = {
    "white_ell": get_white(jnp.array(0.05 * scale), jnp.log(jnp.array(0.05 * scale)), X, ell=0.2 * scale, sigma=1.0),
    "white_sigma": get_white(jnp.array(0.3), jnp.log(jnp.array(0.3)), X, ell=0.2 * scale, sigma=1.0),
    "white_omega": get_white(jnp.array(0.05), jnp.log(jnp.array(0.05)), X, ell=0.3 * scale, sigma=1.0),
    "white_ell_mean": jnp.log(jnp.array(0.05 * scale)),
    "white_sigma_mean": jnp.log(jnp.array(0.3)),
    "white_omega_mean": jnp.log(jnp.array(0.05)),
    "ell_gp_log_ell": jnp.log(jnp.array(0.2 * scale)),
    "sigma_gp_log_ell": jnp.log(jnp.array(0.2 * scale)),
    "omega_gp_log_ell": jnp.log(jnp.array(0.3 * scale)),
    "ell_gp_log_sigma": jnp.log(jnp.array(1.0)),
    "sigma_gp_log_sigma": jnp.log(jnp.array(1.0)),
    "omega_gp_log_sigma": jnp.log(jnp.array(1.0)),
}


value_and_grad_fn = partial(value_and_grad_fn, X=X, y=y)
print("Initial loss", value_and_grad_fn(params))
# sys.exit()

time_it("Setup done")

result = train_fn(value_and_grad_fn, params, optax.adam(0.01), n_iters=2000)
# res = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=value_and_grad_fn).run(params)
# result = {"raw_params": res.params}

time_it("Training done")

plt.figure(figsize=(10, 3))
plt.plot(result["loss_history"])
plt.savefig(f"{data_name}_loss_wdt.png")


fig, ax = plt.subplots(1, 1, figsize=(15, 3))
time_it("Plotting loss done")

print(value_and_grad_fn(result["raw_params"]))

pred_mean, pred_cov, pred_noise = predict_fn(result["raw_params"], X, y, X_test)
time_it("Prediction done")

ax.scatter(X, y, label="data")
ax.plot(X_test, pred_mean, label="mean")
ax.fill_between(
    X_test[:, 0],  # x
    pred_mean - 2 * jnp.sqrt(pred_cov.diagonal()),  # y1
    pred_mean + 2 * jnp.sqrt(pred_cov.diagonal()),  # y2
    alpha=0.5,
    label="2 std",
)
ax.fill_between(
    X_test[:, 0],  # x
    pred_mean - 2 * jnp.sqrt(pred_cov.diagonal() + pred_noise),  # y1
    pred_mean + 2 * jnp.sqrt(pred_cov.diagonal() + pred_noise),  # y2
    alpha=0.5,
    label="2 std + noise",
)
ax.legend()
fig.savefig(f"{data_name}_posterior_wdt.png")

print("ell lgp", jnp.exp(result["raw_params"]["ell_gp_log_ell"]))
print("ell sgp", jnp.exp(result["raw_params"]["sigma_gp_log_ell"]))
print("ell ogp", jnp.exp(result["raw_params"]["omega_gp_log_ell"]))
print("sigma lgp", jnp.exp(result["raw_params"]["ell_gp_log_sigma"]))
print("sigma sgp", jnp.exp(result["raw_params"]["sigma_gp_log_sigma"]))
print("sigma ogp", jnp.exp(result["raw_params"]["omega_gp_log_sigma"]))

fig, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(
    X_test,
    predict_h(
        result["raw_params"]["white_omega"],
        result["raw_params"]["white_omega_mean"],
        X,
        X_test,
        ell=jnp.exp(result["raw_params"]["omega_gp_log_ell"]),
        sigma=jnp.exp(result["raw_params"]["omega_gp_log_sigma"]),
    )[1],
    label="omega",
)
ax.plot(
    X_test,
    predict_h(
        result["raw_params"]["white_ell"],
        result["raw_params"]["white_ell_mean"],
        X,
        X_test,
        ell=jnp.exp(result["raw_params"]["ell_gp_log_ell"]),
        sigma=jnp.exp(result["raw_params"]["ell_gp_log_sigma"]),
    )[1],
    label="ell",
)
ax.plot(
    X_test,
    predict_h(
        result["raw_params"]["white_sigma"],
        result["raw_params"]["white_sigma_mean"],
        X,
        X_test,
        ell=jnp.exp(result["raw_params"]["sigma_gp_log_ell"]),
        sigma=jnp.exp(result["raw_params"]["sigma_gp_log_sigma"]),
    )[1],
    label="sigma",
)
ax.set_ylim(0, 2)
ax.legend()

fig.savefig(f"{data_name}_latent_fn_wdt.png")

time_it("Plotting done")

print()
