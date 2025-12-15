from __future__ import annotations

import numpy as np

from .utils import get_norm_enum, same_dtype_or_raise
from . import chi2_cuda_helper

__all__ = ["lombscargle"]


def lombscargle(
    t,
    y,
    fmin,
    df,
    Nf,
    dy=None,
    nthreads=None,  # kept for API symmetry; not used in CUDA path
    center_data=True,
    fit_mean=True,
    normalization="standard",
    verbose=False,
    cufinufft_kwargs=None,
    nterms=1,
    tpb=None,
):
    """
    CUDA Lomb-Scargle fast chi-squared using a single CUDA helper call.
    Preprocess, NUFFT (cufinufft C API), and postprocess all run on GPU.
    """
    if nterms == 0 and not fit_mean:
        raise ValueError("Cannot have nterms = 0 without fitting bias")

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

    if dy.size == 1:
        dy = np.full_like(y, dy.item())

    Nbatch, _ = y.shape

    default_cufinufft_kwargs = dict(eps="default", gpu_method=1)
    cufinufft_kwargs = {**default_cufinufft_kwargs, **(cufinufft_kwargs or {})}
    if cufinufft_kwargs["eps"] == "default":
        cufinufft_kwargs["eps"] = 1e-5 if dtype == np.float32 else 1e-9
    eps = float(cufinufft_kwargs["eps"])
    gpu_method = int(cufinufft_kwargs.get("gpu_method", 1))
    block_dim = -1 if tpb is None else int(tpb)

    norm_kind = get_norm_enum(normalization)

    power = chi2_cuda_helper.lombscargle_chi2_cuda(
        t.astype(dtype, copy=False),
        y.astype(dtype, copy=False),
        dy.astype(dtype, copy=False),
        dtype.type(fmin),
        dtype.type(df),
        int(Nf),
        bool(center_data),
        bool(fit_mean),
        norm_kind,
        int(nterms),
        eps,
        gpu_method,
        block_dim,
    )

    if squeeze_output:
        power = power.squeeze()

    if verbose:
        print(
            f"[nifty-ls cufinufft_chi2_CUDA] batches={Nbatch}, dtype={dtype}, eps={eps}, nterms={nterms}"
        )

    return power
