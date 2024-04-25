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
