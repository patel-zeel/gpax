* Global methods to set jitter, positive bijector, default bijector are kept to globally experiment with different settings.

* We included setting priors and constraints within __initialize_params__ to allow __post_initialize_params__ to function correctly. If this is not done, state of self.priors and self.constraints can not be efficiently managed during multiple calls of __initialize_params__. See HeteroscedasticGaussian likelihood for example.
    * self.priors and self.constrains must be defined after params to allow recursive calls to child modules.

* \ell, \sigma and \omega should be modelled without squaring them. This generelizes latent GPs that they are all modelling similar parameters.

# New idea
In case of inducing points methods, p(l) can be computed by solving integral p(l|l_bar)p(l_bar)dl_bar which would be equivalent to vanilla heinonen approach.