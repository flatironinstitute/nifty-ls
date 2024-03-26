from __future__ import annotations

import numpy as np


def validate_frequency_grid(
    fmin, fmax, Nf, t, assume_sorted_t=True, samples_per_peak=5, nyquist_factor=5
):
    """
    Validate the frequency grid parameters and return them in a canonical form.
    """

    # TODO: the problem is overspecified, we may need to rethink the interface

    if fmin is None or fmax is None or Nf is None:
        if assume_sorted_t:
            tmin = t[0]
            tmax = t[-1]
        else:
            tmin = np.min(t)
            tmax = np.max(t)

        baseline = tmax - tmin

        if fmax is None:
            avg_nyquist = 0.5 * len(t) / baseline
            fmax = avg_nyquist * nyquist_factor

        if fmin is None:
            fmin = 1 / (2 * samples_per_peak * baseline)

        if Nf is None:
            Nf = 1 + int(np.round((fmax - fmin) / df))  # noqa: F821

    fmin = float(fmin)
    fmax = float(fmax)
    Nf = int(Nf)

    if fmin >= fmax:
        raise ValueError('fmin must be less than fmax')

    if Nf < 1:
        raise ValueError('Nf must be a positive integer')

    df = (fmax - fmin) / (Nf - 1)  # fmax inclusive

    return fmin, df, Nf
