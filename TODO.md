* X_inducing should be available with GP object only
* Combine Heinonen and our version of GPs
    * Our version: Set include_prior to False and set prior for latent GP hyperparams
    * Heinonen version: In Exact GP, 
* Create separate model files for models and use this library as core.

# Thoughts
- Support classification by generalizing likelihood?
- Want to support separate X_inducing for GP, kernel and noise?

# Pending
- Enable an option outside to train i) latent gp noise or not ii) latent gp hyperparams or not
- Add mu also during reparametrization.