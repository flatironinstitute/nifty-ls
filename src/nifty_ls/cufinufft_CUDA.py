"""
CUDA Lomb-Scargle periodogram using gpu_helper (nanobind CUDA kernels + cufinufft C API).
This variant avoids CuPy entirely: inputs are NumPy arrays, and all device allocation
and transfers are handled inside the C++/CUDA binding.
"""

from __future__ import annotations

import numpy as np

from .utils import get_norm_enum, same_dtype_or_raise
from . import gpu_helper

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
):
    """
    Run Lomb-Scargle on GPU using cufinufft C API without requiring CuPy on the Python side.

    Parameters
    ----------
    t, y, dy : array-like
        1-D time array and 1- or 2-D values/uncertainties (shape (Nbatch, N) or (N,)).
    fmin, df : float
        Minimum frequency and frequency step (radians per unit time).
    Nf : int
        Number of frequency bins.
    center_data : bool
        Subtract weighted mean before transform.
    fit_mean : bool
        Fit floating mean term (requires two NUFFTs per batch).
    normalization : str
        "standard" or "psd".
    verbose : bool
        Print timings (host-side only).
    cufinufft_kwargs : dict
        Supports "eps" and "gpu_method"; defaults eps to 1e-5 (float32) / 1e-9 (float64).
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

    power = gpu_helper.lombscargle_gpu(
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
    )

    if squeeze_output:
        power = power.squeeze()

    if verbose:
        print(f"[nifty-ls cufinufft_CUDA] batches={Nbatch}, dtype={dtype}, eps={eps}")

    return power
