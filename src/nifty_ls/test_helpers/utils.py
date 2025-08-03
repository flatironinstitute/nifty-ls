"""nifty-ls test helpers"""

from __future__ import annotations

import numpy as np

from ..utils import validate_frequency_grid, validate_frequency_grid_mp


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


def gen_data_mp(N_series=100_000, N_batch=None, N_d=100, dtype=np.float64, seed=5043):
    rng = np.random.default_rng(seed)

    # allow lengths from 50% to 150% of N_d
    min_len = max(1, int(N_d * 0.5))
    max_len = int(N_d * 1.5)

    t_list = []
    y_list = []
    dy_list = []

    N_batch = N_batch if N_batch else 1

    for _ in range(N_series):
        # random series length
        N_d_i = rng.integers(min_len, max_len + 1)
        t_i = np.sort(rng.random(N_d_i, dtype=dtype)) * 123

        if N_batch:
            freqs = rng.random((N_batch, 1), dtype=dtype) * 10 + 1
            # broadcast over time: (N_batch, N_d_i)
            y_i = np.sin(freqs * t_i) + 1.23
            dy_i = rng.random((N_batch, N_d_i), dtype=dtype) * 0.1 + 0.01
            noise = rng.normal(0, dy_i, size=(N_batch, N_d_i))
            y_i = y_i + noise

        # make read-only
        t_i.setflags(write=False)
        y_i.setflags(write=False)
        dy_i.setflags(write=False)
        
        t_list.append(t_i)
        y_list.append(y_i)
        dy_list.append(dy_i)

    fmin_list, df_list, Nf_list = validate_frequency_grid_mp(
        fmin=None, fmax=None, Nf=None, t_list=t_list)

    fmax_list = [fmin_list[i] + df_list[i] * (Nf_list[i] - 1) for i in range(len(fmin_list))]

    return dict(t=t_list, y=y_list, dy=dy_list, fmin=fmin_list, fmax=fmax_list, df=df_list, Nf=Nf_list)


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
