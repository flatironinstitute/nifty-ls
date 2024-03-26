from __future__ import annotations

import importlib

import numpy as np

from . import utils
from . import backends


AVAILABLE_BACKENDS = backends.available_backends()


def lombscargle(
    t,
    y,
    dy,
    fmin=None,
    fmax=None,
    Nf=None,
    center_data: bool = True,
    fit_mean: bool = True,
    normalization: str = 'standard',
    assume_sorted_t: bool = True,
    samples_per_peak=5,
    nyquist_factor=5,
    backend='finufft',
    backend_kwargs=None,
):
    """
    Parameters
    ----------
    t : array_like or list of array_like
        Times of observations. If `t` is an array, then it must be one dimensional (shape (N,)).
        If `t` is a list of arrays, then each of the arrays must be one dimensional,
        but they can have different lengths (with `t[i]` having shape `(N_i,)`).
    y : array_like or list of array_like
        Measurement values. If `y` is an array, then it must have shape `([M,] N)`, where `M` is the number of
        light curves sharing the same observation times and `N` is the number of observations.
        If `y` is a list of arrays, then each of the arrays must be one dimensional,
        but they can have different lengths (`y[i]` must have shape `(N_i,)`).
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
        **(backend_kwargs or {}),
    )

    return dict(power=power, fmin=fmin, df=df, Nf=Nf)


def lombscargle_freq(nifty_result):
    return nifty_result['fmin'] + nifty_result['df'] * np.arange(nifty_result['Nf'])
