from __future__ import annotations


import numpy as np

__all__ = ['validate_frequency_grid']


def validate_frequency_grid(
    fmin, fmax, Nf, t, assume_sorted_t=True, samples_per_peak=5, nyquist_factor=5
):
    """
    Validate the frequency grid parameters and return them in a canonical form.
    Follows the Astropy LombScargle conventions for samples_per_peak and nyquist_factor.
    """

    if fmin is None or fmax is None or Nf is None:
        if assume_sorted_t:
            baseline = t[-1] - t[0]
        else:
            baseline = np.ptp(t)

        if baseline <= 0.0:
            raise ValueError(
                'The input time array must be non-degenerate, '
                'and sorted if assume_sorted_t=True.'
            )

        target_df = 1 / (samples_per_peak * baseline)

        if fmax is None:
            avg_nyquist = 0.5 * len(t) / baseline
            fmax = avg_nyquist * nyquist_factor

        if fmin is None:
            fmin = target_df / 2

        if Nf is None:
            Nf = 1 + int(np.round((fmax - fmin) / target_df))

    fmin = float(fmin)
    fmax = float(fmax)
    Nf = int(Nf)

    if fmin >= fmax:
        raise ValueError('fmin must be less than fmax')

    if Nf < 1:
        raise ValueError('Nf must be a positive integer')

    df = (fmax - fmin) / (Nf - 1)  # fmax inclusive

    return fmin, df, Nf


def same_dtype_or_raise(**arrays):
    """
    Check if all arrays have the same dtype, raise ValueError if not.
    """
    dtypes = {n: a.dtype for (n, a) in arrays.items() if a is not None}
    names = list(dtypes.keys())

    for n in names[1:]:
        if dtypes[n] != dtypes[names[0]]:
            raise ValueError(
                f'Arrays {names[0]} and {n} have different dtypes: '
                f'{dtypes[names[0]]} and {dtypes[n]}'
            )
