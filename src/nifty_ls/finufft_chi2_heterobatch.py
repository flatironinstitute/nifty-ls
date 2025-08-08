from __future__ import annotations

__all__ = ['lombscargle_heterobatch']

import numpy as np

from nifty_ls.finufft_chi2_heterobatch_helpers import process_chi2_hetero_batch
from nifty_ls.finufft import FFTW_ESTIMATE
from .utils import same_dtype_or_raise, broadcast_dy_list
from .finufft import get_finufft_max_threads


def lombscargle_heterobatch(
    t_list,
    y_list,
    fmin_list,
    df_list,
    Nf_list,
    dy_list=None,
    nthreads=None,
    center_data=True,
    fit_mean=True,
    normalization='standard',
    eps='default',
    upsampfac=1.25,
    fftw=FFTW_ESTIMATE,
    nterms=1,
    verbose=False,
):
    """
    Compute multiple series of Chi2 Lomb-Scargle periodogram with heterogeneous lengths.

    This function extends the heterobatch capability to the Chi2 method, allowing fitting
    of multiple harmonic terms on time series of different lengths.

    Parameters
    ----------
    t_list : List of array-like
        The time values, shape (N_series, N_d_i) for i in [0..N_series-1]
    y_list : List of array-like
        The data values, shape (N_series, N_y_i, N_t_i) or (N_series, N_t_i)
        for i in [0..N_series-1].
    fmin_list : List of float
        The minimum frequency for each series.
    df_list : List of float
        The frequency bin width for each series.
    Nf_list : List of int
        The number of frequency bins for each series.
    dy_list : List of array-like, optional
        Measurement uncertainties for the data values. Can be provided as:
        - A single scalar (uniform uncertainty for all data points across all series)
        - A list of scalars (one uncertainty value per series)
        - A list of arrays (element-wise uncertainties matching shapes of corresponding y_list entries)
    nthreads : int, optional
        The number of threads to use. Default is to use maximum available threads.
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data. Default is True.
    normalization : str, optional
        The normalization method. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    eps : float or str, optional
        The precision tolerance for FINUFFT. Default is 'default', which uses 1e-5 for float32
        and 1e-9 for float64.
    upsampfac : float, optional
        The upsampling factor for FINUFFT. Larger values improve accuracy at the cost of speed.
        Default is 1.25.
    fftw : int, optional
        The FFTW planning flag to use. Options are FFTW_ESTIMATE or FFTW_MEASURE.
        Default is FFTW_ESTIMATE.
    verbose : bool, optional
        Whether to print verbose output during computation. Default is False.
    nterms : int, optional
        Number of Fourier terms in the fit. Default is 1.

    Returns
    -------
    powers : List of array-like
        A list of computed periodogram arrays, one for each input series.
    """

    if nterms == 0 and not fit_mean:
        raise ValueError('Cannot have nterms = 0 without fitting bias')

    # Use max threads if not specified
    if nthreads is None:
        nthreads = get_finufft_max_threads()

    # N_series and data type check
    N_series = len(t_list)

    # Verify variable sizes and dtypes
    if len(y_list) != N_series:
        raise ValueError('Time series(t), observation(y) should have same length')
    broadcased_dy_list = broadcast_dy_list(y_list=y_list, dy_list=dy_list)
    for i in range(N_series):
        same_dtype_or_raise(
            t=t_list[i],
            y=y_list[i],
            dy=broadcased_dy_list[i] if broadcased_dy_list else broadcased_dy_list,
        )
    dtype = t_list[0].dtype

    # Set precision based on dtype
    if eps == 'default':
        if dtype == np.float32:
            eps = 1e-5
        else:
            eps = 1e-9

    # Pre-allocate space for powers
    powers = []
    for i in range(len(t_list)):
        # Make sure y is at least 2D
        y_shape = y_list[i].shape
        if len(y_shape) == 1:
            y_list[i] = y_list[i].reshape(1, -1)
            y_shape = y_list[i].shape

        power = np.empty((y_shape[0], Nf_list[i]), dtype=dtype)
        powers.append(power)

    # Call C++ function
    process_chi2_hetero_batch(
        t_list,
        y_list,
        broadcased_dy_list,
        fmin_list,
        df_list,
        Nf_list,
        powers,
        normalization,
        nthreads,
        center_data,
        fit_mean,
        eps,
        upsampfac,
        fftw,
        nterms,
        verbose,
    )

    return powers
