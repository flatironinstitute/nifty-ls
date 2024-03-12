"""
Hooks for astropy's Lomb-Scargle implementation.
"""

from .core import lombscargle_finufft


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
    nthreads=1,
    **finufft_kwargs,
):
    """
    This will always use finufft/cufinufft, rather than a brute-force implementation.
    Brute-force should only be used with many small time series, which needs to go
    through the nifty-ls interface directly because Astropy doesn't support batching.
    """

    return lombscargle_finufft(
        t,
        y,
        dy,
        fmin=f0,
        df=df,
        Nf=Nf,
        # center_data=center_data,
        # fit_mean=fit_mean,  # TODO
        normalization=normalization,
        nthreads=nthreads,
        **finufft_kwargs,
    )
