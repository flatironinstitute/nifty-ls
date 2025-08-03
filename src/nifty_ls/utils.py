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
        raise ValueError("t_list must be provided as a list of time arrays.")
    N_series = len(t_list)

    # Helper to broadcast scalars to lists
    def _broadcast(x, name, cast_fn):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != N_series:
                raise ValueError(f"Length of '{name}' must match number of series ({N_series}).")
            return list(x)
        else:
            return [x] * N_series
    
    fmin_vals = _broadcast(fmin, "fmin", float)
    fmax_vals = _broadcast(fmax, "fmax", float)
    Nf_vals   = _broadcast(Nf,   "Nf",   int)

    fmin_list = []
    df_list = []
    Nf_list = []

    baseline_list = []
    for ti in t_list:
        if ti.size < 2:
            raise ValueError("Each time array must have at least two points.")
        b = (ti[-1] - ti[0]) if assume_sorted_t else np.ptp(ti)
        if b <= 0:
            raise ValueError("Time array must be non-degenerate and sorted if assume_sorted_t=True.")
        baseline_list.append(float(b))
    
    for i, ti in enumerate(t_list):
        baseline = baseline_list[i]
        target_df = 1.0 / (samples_per_peak * baseline)

        if fmax_vals[i] is None:
            avg_nyquist = 0.5 * ti.size / baseline
            fmax_i = avg_nyquist * nyquist_factor
        else:
            fmax_i = float(fmax_vals[i])

        if fmin_vals[i] is None:
            fmin_i = target_df / 2.0
        else:
            fmin_i = float(fmin_vals[i])

        if Nf_vals[i] is None:
            Nf_i = 1 + int(np.round((fmax_i - fmin_i) / target_df))
        else:
            Nf_i = int(Nf_vals[i])

        if fmin_i >= fmax_i:
            raise ValueError(f"fmin({fmin_i}) ≥ fmax({fmax_i}) at index {i}.")
        
        if Nf_i < 1:
            raise ValueError(f"Nf at index {i} must be positive, got {Nf_i}.")
        
        df_i = (fmax_i - fmin_i) / (Nf_i - 1)
        if df_i <= 0:
            raise ValueError(f"Computed df({df_i}) must be positive at index {i}.")
        
        fmin_list.append(fmin_i)
        df_list.append(df_i)
        Nf_list.append(Nf_i)

    return fmin_list, df_list, Nf_list

def same_dtype_or_raise(**arrays):
    """
    Check if all arrays have the same dtype, raise ValueError if not.
    """
    dtypes = {n: a.dtype for (n, a) in arrays.items() if a is not None 
              and not np.isscalar(a)}
    names = list(dtypes.keys())

    for n in names[1:]:
        if dtypes[n] != dtypes[names[0]]:
            raise ValueError(
                f'Arrays {names[0]} and {n} have different dtypes: '
                f'{dtypes[names[0]]} and {dtypes[n]}'
            )
