"""
Hooks for astropy's Lomb-Scargle implementation.  The hooks are installed in __init__.py.
"""

from __future__ import annotations

from .core import lombscargle


def lombscargle_fastnifty(
    t,
    y,
    dy,
    f0,
    df,
    Nf,
    center_data=True,
    fit_mean=True,
    normalization='standard',
    **lombscargle_kwargs,
):
    """
    Usually one will want to use the finufft or cufinufft backends, rather than a brute-force backend.
    Brute-force is probably only faster for many small NUFFTs, for which one needs to use the native
    nifty-ls interface anyway (because astropy does not support batching).
    """

    return lombscargle(
        t,
        y,
        dy,
        fmin=f0,
        fmax=f0 + df * (Nf - 1),
        Nf=Nf,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        **lombscargle_kwargs,
    )['power']
