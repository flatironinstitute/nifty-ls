import logging

from .core import lombscargle
from .core import NiftyResult

__all__ = ['lombscargle', 'NiftyResult']

# Make "fastnifty" available as a method for astropy's Lomb Scargle
try:
    import astropy.timeseries.periodograms.lombscargle.implementations.main as astropy_ls
except ImportError:
    logging.info('Astropy not found, fastnifty method will not be available')
    astropy_ls = None

if astropy_ls:
    from .astropy import lombscargle_fastnifty

    astropy_ls.METHODS['fastnifty'] = lombscargle_fastnifty
