from ._version import version as __version__  # noqa

import os

# defaults
from gpax.defaults import set_default_jitter

# distributions
from gpax.distributions import Normal
from gpax.distributions import set_default_prior

# bijectors
from gpax.bijectors import Identity, Exp, SquarePlus
from gpax.bijectors import set_default_bijector, set_positive_bijector

## Set defaults
DEFAULT_JITTER = 1e-6
set_default_prior(Normal)
set_default_bijector(Identity)
set_positive_bijector(SquarePlus)
set_default_jitter(DEFAULT_JITTER)
