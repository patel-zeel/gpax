from ._version import version as __version__  # noqa

import os

# defaults
from gpax.defaults import set_default_jitter

# distributions
from gpax.distributions import Normal
from gpax.distributions import set_default_prior

# bijectors
import gpax.bijectors as gb
from gpax.bijectors import set_default_bijector, set_positive_bijector

## Set defaults
DEFAULT_JITTER = 1e-6
set_default_prior(Normal)
set_default_bijector(gb.Identity)
set_positive_bijector(gb.Exp)
set_default_jitter(DEFAULT_JITTER)
