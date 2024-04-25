from __future__ import annotations

import importlib
from typing import Optional, Literal

import numpy as np
import numpy.typing as npt

from . import utils
from .backends import available_backends, BACKEND_TYPE

__all__ = [
    'lombscargle',
    'lombscargle_freq',
    'NORMALIZATION_TYPE',
    'AVAILABLE_BACKENDS',
]


AVAILABLE_BACKENDS = available_backends()
NORMALIZATION_TYPE = Literal['standard', 'model', 'log', 'psd']


def lombscargle(
    t: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    dy: npt.NDArray[np.floating],
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    Nf: Optional[int] = None,
    center_data: bool = True,
    fit_mean: bool = True,
    normalization: NORMALIZATION_TYPE = 'standard',
    assume_sorted_t: bool = True,
    samples_per_peak: int = 5,
    nyquist_factor: int = 5,
    backend: BACKEND_TYPE = 'finufft',
    **backend_kwargs: Optional[dict],
) -> dict:
    """
    Compute a Lomb-Scargle periodogram, or a batch of periodograms if `y` and `dy` are 2D arrays.

    This function can dispatch to multiple backends, including 'finufft' and 'cufinufft'. The latter
    uses CUDA and requires that nifty-ls was installed with the 'cuda' extra.

    The result is a dictionary containing the computed periodograms, as well as the frequency grid parameters.
    The actual frequency grid can be obtained by passing the result dict to `lombscargle_freq()`.

    The meanings of these parameters conform to the Lomb-Scargle implementation in Astropy:
    https://docs.astropy.org/en/stable/timeseries/lombscargle.html

    Parameters
    ----------
    t : array-like
        The time values, shape (N_t,)
    y : array-like
        The data values, shape (N_t,) or (N_y, N_t)
    dy : array-like
        The uncertainties of the data values, broadcastable to `y`
    fmin : float, optional
        The minimum frequency of the periodogram. If not provided, it will be chosen automatically.
    fmax : float, optional
        The maximum frequency of the periodogram. If not provided, it will be chosen automatically.
    Nf : int, optional
        The number of frequency bins. If not provided, it will be chosen automatically.
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data before computing the periodogram. Default is True.
    normalization : str, optional
        The normalization method to use. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    assume_sorted_t : bool, optional
        Whether to assume that the time values are sorted in ascending order, allowing for a performance
        optimization in determining the frequency grid.  Default is True.
    samples_per_peak : int, optional
        The number of samples per peak to use when determining the frequency grid. Default is 5.
    nyquist_factor : int, optional
        The factor by which to multiply the Nyquist frequency when determining the frequency grid. Default is 5.
    backend : str, optional
        The backend to use for the computation. Default is 'finufft'.
    backend_kwargs : dict, optional
        Additional keyword arguments to pass to the backend.

    Returns
    -------
    nifty_result : dict
        A dictionary containing the computed periodogram(s), as well as the frequency grid parameters.
        The keys are 'power', 'fmin', 'df', 'Nf', and 'fmax'.
        nifty_result['power'] will be an ndarray of shape (Nf,) or (N_y, Nf) if `y` is 2D.
    """
    fmin, df, Nf = utils.validate_frequency_grid(
        fmin,
        fmax,
        Nf,
        t,
        assume_sorted_t=assume_sorted_t,
        samples_per_peak=samples_per_peak,
        nyquist_factor=nyquist_factor,
    )

    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f'Unknown or unavailable backend: {backend}. Available backends are: {AVAILABLE_BACKENDS}'
        )

    backend_module = importlib.import_module(f'.{backend}', __package__)

    power = backend_module.lombscargle(
        t=t,
        y=y,
        dy=dy,
        fmin=fmin,
        df=df,
        Nf=Nf,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        **backend_kwargs,
    )

    fmax = fmin + df * (Nf - 1)
    nifty_result = dict(power=power, fmin=fmin, df=df, Nf=Nf, fmax=fmax)

    return nifty_result


def lombscargle_freq(nifty_result):
    """
    Return the frequency grid corresponding to the result of `lombscargle()`.

    Parameters
    ----------
    nifty_result : dict
        The result of a call to `lombscargle`, containing the keys 'fmin', 'df', and 'Nf'.

    Returns
    -------
    freq : ndarray
        The frequency grid.
    """

    return nifty_result['fmin'] + nifty_result['df'] * np.arange(nifty_result['Nf'])
