from __future__ import annotations

__all__ = ['lombscargle_heterobatch']

import numpy as np

from nifty_ls.finufft_heterobatch_helpers import process_hetero_batch
from nifty_ls.finufft import FFTW_ESTIMATE
from .utils import same_dtype_or_raise, broadcast_dy_list, get_norm_enum
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
    verbose=False,
):
    """
    Compute multiple series of Lomb-Scargle periodogram, or a batch of periodograms if `y` and `dy` are 2D arrays.

    This function can dispatch to multiple backends, including 'finufft_heterobatch'. Plan to implement CUDA
    version latter.

    The meanings of these parameters conform to the Lomb-Scargle implementation in Astropy:
    https://docs.astropy.org/en/stable/timeseries/lombscargle.html

    Parameters
    ----------
    t_list : list of array-like
        The time values, shape (N_series, N_d_i) for i in [0..N_series-1]
    y_list : list of array-like
        The data values, shape (N_series, N_t_i) or (N_series, N_y, N_t_i)
        for i in [0..N_series-1].
    fmin_list : list of float, optional
        The minimum frequency of the periodogram. If not provided, it will be chosen automatically.
    df_list : list of float, optional
        The frequency bin width of the periodogram.
    Nf_list : list of int, optional
        The number of frequency bins. If not provided, it will be chosen automatically.
    dy_list : list of array-like, optional
        List of the uncertainties of the data values, broadcastable to `y`.
    nthreads : int, optional
        The number of threads to use. The default behavior is to use maximum threads
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data before computing the periodogram. Default is True.
    normalization : str, optional
        The normalization method to use. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    eps : float or str, optional
        The precision tolerance for FINUFFT. Default is 'default', which uses 1e-5 for float32 and 1e-9 for float64.
    upsampfac : float, optional
        The upsampling factor for the FFT used in FINUFFT. Larger values improve accuracy at the cost of speed.
        Default is 1.25.
    fftw : int, optional
        The FFTW planning flag to use. Options are FFTW_ESTIMATE (faster planning, slower FFT)
        or FFTW_MEASURE (slower planning, faster FFT). Default is FFTW_ESTIMATE.
    verbose : bool, optional
        Whether to print verbose output during computation. Default is False.

    Returns
    -------
    power: list of npt.NDArray[np.floating]
    """

    # Use max threads
    if nthreads is None:
        nthreads = get_finufft_max_threads()

    # N_series and data type check
    N_series = len(t_list)

    # Verify variable sizes and dtypes
    if len(y_list) != N_series:
        raise ValueError('Time series(t), observation(y) should have same length')
    dy_list = broadcast_dy_list(y_list=y_list, dy_list=dy_list)
    for i in range(N_series):
        same_dtype_or_raise(
            t=t_list[i],
            y=y_list[i],
            dy=dy_list[i] if i < len(dy_list) else None,
        )
    dtype = t_list[0].dtype

    if eps == 'default':
        if dtype == np.float32:
            eps = 1e-5
        else:
            eps = 1e-9

    normalization = get_norm_enum(normalization)

    squeeze_output = [y.ndim == 1 for y in y_list]
    y_list = [np.atleast_2d(y) for y in y_list]
    dy_list = [np.atleast_2d(dy) for dy in dy_list]

    # Pre-allocate space for powers
    powers = []
    for i in range(len(t_list)):
        power = np.empty((y_list[i].shape[0], Nf_list[i]), dtype=dtype)
        powers.append(power)

    # Call C++ function
    process_hetero_batch(
        t_list,
        y_list,
        dy_list,
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
        verbose,
    )

    for i in range(len(powers)):
        if squeeze_output[i]:
            powers[i] = np.squeeze(powers[i], axis=0)

    return powers
