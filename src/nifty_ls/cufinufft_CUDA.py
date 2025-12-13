"""
CUDA Lomb-Scargle periodogram using cuda_helper (nanobind CUDA kernels + cufinufft C API).
This variant avoids CuPy entirely: inputs are NumPy arrays, and all device allocation
and transfers are handled inside the C++/CUDA binding.
"""

from __future__ import annotations

import numpy as np

from .utils import get_norm_enum, same_dtype_or_raise
from . import cuda_helper

__all__ = ["lombscargle"]


def lombscargle(
    t,
    y,
    fmin,
    df,
    Nf,
    dy=None,
    center_data=True,
    fit_mean=True,
    normalization="standard",
    verbose=False,
    cufinufft_kwargs=None,
    tpb=None,
):
    """
    Run Lomb-Scargle on GPU using cufinufft C API and CUDA kernels via nanobind.

    Parameters
    ----------
    t : array-like
        The time values, shape (N_t,)
    y : array-like
        The data values, shape (N_t,) or (N_y, N_t)
    fmin : float
        The minimum frequency of the periodogram.
    df : float
        The frequency bin width.
    Nf : int
        The number of frequency bins.
    dy : array-like, optional
        The uncertainties of the data values, broadcastable to `y`
    nthreads : int, optional
        The number of threads to use. The default behavior is to use (N_t / 4) * (Nf / 2^15) threads,
        capped to the maximum number of OpenMP threads. This is a heuristic that may not work well in all cases.
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data before computing the periodogram. Default is True.
    normalization : str, optional
        The normalization method to use. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    _no_cpp_helpers : bool, optional
        Whether to use the pure Python implementation of the finufft pre- and post-processing.
        Default is False.
    verbose : bool, optional
        Whether to print additional information about the finufft computation.
    finufft_kwargs : dict, optional
        Additional keyword arguments to pass to the `finufft.Plan()` constructor.
        Particular finufft parameters of interest may be:
        - `eps`: the requested precision [1e-9 for double precision and 1e-5 for single precision]
        - `upsampfac`: the upsampling factor [1.25]
        - `fftw`: the FFTW planner flags [FFTW_ESTIMATE]
    """

    same_dtype_or_raise(t=t, y=y, dy=dy)

    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy) if dy is not None else None

    dtype = t.dtype
    if dtype not in (np.float32, np.float64):
        raise TypeError("t/y/dy must be float32 or float64")

    if dy is None:
        dy = np.array(1.0, dtype=dtype)

    squeeze_output = y.ndim == 1
    y = np.atleast_2d(y)
    dy = np.atleast_2d(dy)

    # Broadcast dy if provided as scalar
    if dy.size == 1:
        dy = np.full_like(y, dy.item())

    Nbatch, _ = y.shape

    default_cufinufft_kwargs = dict(eps="default", gpu_method=1)
    cufinufft_kwargs = {**default_cufinufft_kwargs, **(cufinufft_kwargs or {})}

    if cufinufft_kwargs["eps"] == "default":
        cufinufft_kwargs["eps"] = 1e-5 if dtype == np.float32 else 1e-9

    eps = float(cufinufft_kwargs["eps"])
    gpu_method = int(cufinufft_kwargs.get("gpu_method", 1))

    norm_kind = get_norm_enum(normalization)
    block_dim = -1 if tpb is None else int(tpb)

    power = cuda_helper.lombscargle_cuda(
        t.astype(dtype, copy=False),
        y.astype(dtype, copy=False),
        dy.astype(dtype, copy=False),
        dtype.type(fmin),
        dtype.type(df),
        int(Nf),
        bool(center_data),
        bool(fit_mean),
        norm_kind,
        eps,
        gpu_method,
        block_dim,
    )

    if squeeze_output:
        power = power.squeeze()

    if verbose:
        print(f"[nifty-ls cufinufft_CUDA] batches={Nbatch}, dtype={dtype}, eps={eps}")

    return power
