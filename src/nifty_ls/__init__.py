import logging

from .core import lombscargle
from .core import NiftyResult

from .version import __version__

__all__ = ['lombscargle', 'NiftyResult', '__version__']


# Make "fastnifty" and "fastnifty_chi2" available as a method for astropy's Lomb Scargle
try:
    import astropy.timeseries.periodograms.lombscargle.implementations.main as astropy_ls
except ImportError:
    logging.info('Astropy not found, fastnifty method will not be available')
    astropy_ls = None

if astropy_ls:
    from .astropy import lombscargle_fastnifty, lombscargle_fastnifty_chi2

    astropy_ls.METHODS['fastnifty'] = lombscargle_fastnifty
    astropy_ls.METHODS['fastnifty_chi2'] = lombscargle_fastnifty_chi2
