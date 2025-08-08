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


def validate_frequency_grid_mp(
    fmin, fmax, Nf, t_list, assume_sorted_t=True, samples_per_peak=5, nyquist_factor=5
):
    if t_list is None:
        raise ValueError('t_list must be provided as a list of time arrays.')
    N_series = len(t_list)

    # Helper to broadcast scalars to lists
    def _broadcast(x, name, cast_fn):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != N_series:
                raise ValueError(
                    f"Length of '{name}' must match number of series ({N_series})."
                )
            return list(x)
        else:
            return [x] * N_series

    fmin_vals = _broadcast(fmin, 'fmin', float)
    fmax_vals = _broadcast(fmax, 'fmax', float)
    Nf_vals = _broadcast(Nf, 'Nf', int)

    fmin_list = []
    df_list = []
    Nf_list = []

    # Validate each time series individually
    for i, t in enumerate(t_list):
        fmin_i, df_i, Nf_i = validate_frequency_grid(
            fmin_vals[i],
            fmax_vals[i],
            Nf_vals[i],
            t,
            assume_sorted_t=assume_sorted_t,
            samples_per_peak=samples_per_peak,
            nyquist_factor=nyquist_factor,
        )
        fmin_list.append(fmin_i)
        df_list.append(df_i)
        Nf_list.append(Nf_i)

    return fmin_list, df_list, Nf_list


def same_dtype_or_raise(**arrays):
    """
    Check if all arrays have the same dtype, raise ValueError if not.
    """
    dtypes = {
        n: a.dtype for (n, a) in arrays.items() if a is not None and not np.isscalar(a)
    }
    names = list(dtypes.keys())

    for n in names[1:]:
        if dtypes[n] != dtypes[names[0]]:
            raise ValueError(
                f'Arrays {names[0]} and {n} have different dtypes: '
                f'{dtypes[names[0]]} and {dtypes[n]}'
            )


def broadcast_dy_list(y_list, dy_list):
    """
    Handle and check uncertainty values (dy) to match shapes of observation arrays.
    """

    # Handle dy_list is None
    if not dy_list:
        return dy_list

    # Handle scalar
    if np.isscalar(dy_list):
        return None

    # Handle list of scalar
    elif np.isscalar(dy_list[0]):
        if len(dy_list) != len(y_list):
            raise ValueError(
                f'Length mismatch: y_list has {len(y_list)} elements but dy_list has {len(dy_list)} elements'
            )
        for i in range(len(y_list)):
            if not np.isscalar(dy_list[i]):
                raise ValueError(
                    f'Expected scalar uncertainty at index {i}, got {type(dy_list[i]).__name__}'
                )
        return None
    return dy_list
