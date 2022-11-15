# from ._version import version as __version__  # noqa

from gpax.core import set_default_jitter, set_default_prior, set_positive_bijector, set_default_bijector
import gpax.distributions as gd
import gpax.bijectors as gb

## Set defaults
set_default_prior(gd.Normal)
set_default_bijector(gb.Identity)
set_positive_bijector(gb.Exp)

DEFAULT_JITTER = 1e-6

set_default_jitter(DEFAULT_JITTER)
