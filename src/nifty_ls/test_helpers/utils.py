"""nifty-ls test helpers"""

from __future__ import annotations

import numpy as np

from ..utils import validate_frequency_grid


def gen_data(N=100, Nbatch=None, seed=5043, dtype=np.float64):
    rng = np.random.default_rng(seed)

    t = np.sort(rng.random(N, dtype=dtype)) * 123
    freqs = rng.random((Nbatch, 1) if Nbatch else 1, dtype=dtype) * 10 + 1
    y = np.sin(freqs * t) + 1.23
    dy = rng.random(y.shape, dtype=dtype) * 0.1 + 0.01
    y += rng.normal(0, dy, y.shape)

    fmin, df, Nf = validate_frequency_grid(None, None, None, t)
    # The frequency matrix becomes ill-conditioned at the lowest frequencies when nterms > 1,
    # so we increase fmin to mitigate this.
    fmin = 2 * df
    fmax = fmin + df * (Nf - 1)

    t.setflags(write=False)
    y.setflags(write=False)
    dy.setflags(write=False)

    return dict(t=t, y=y, dy=dy, fmin=fmin, fmax=fmax)


def astropy_ls(
    t,
    y,
    dy,
    fmin,
    fmax,
    Nf,
    fit_mean=True,
    center_data=True,
    use_fft=False,
    normalization='standard',
):
    from astropy.timeseries.periodograms.lombscargle.implementations import fast_impl

    df = (fmax - fmin) / (Nf - 1)

    power = fast_impl.lombscargle_fast(
        t,
        y,
        dy,
        fmin,
        df,
        Nf,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        use_fft=use_fft,
    )

    return power


def astropy_ls_fastchi2(
    t,
    y,
    dy,
    fmin,
    fmax,
    Nf,
    fit_mean=True,
    center_data=True,
    use_fft=False,
    normalization='standard',
    nterms=1,
):
    from astropy.timeseries.periodograms.lombscargle.implementations import (
        fastchi2_impl,
    )

    df = (fmax - fmin) / (Nf - 1)

    power = fastchi2_impl.lombscargle_fastchi2(
        t,
        y,
        dy,
        fmin,
        df,
        Nf,
        center_data=center_data,
        fit_mean=fit_mean,
        normalization=normalization,
        use_fft=use_fft,
        nterms=nterms,
    )

    return power
