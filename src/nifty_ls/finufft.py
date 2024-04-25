from __future__ import annotations

import os
from timeit import default_timer as timer

import finufft
import numpy as np

from . import cpu_helpers

FFTW_MEASURE = 0
FFTW_ESTIMATE = 64

MAX_THREADS = len(os.sched_getaffinity(0))

__all__ = ['lombscargle', 'FFTW_MEASURE', 'FFTW_ESTIMATE', 'MAX_THREADS']


def lombscargle(
    t,
    y,
    dy,
    fmin,
    df,
    Nf,
    nthreads=None,
    center_data=True,
    fit_mean=True,
    normalization='standard',
    _no_cpp_helpers=False,
    verbose=False,
    finufft_kwargs=None,
):
    """
    Compute the Lomb-Scargle periodogram using the FINUFFT backend.

    Performance Tuning
    ------------------
    The performance of this backend depends almost entirely on the performance of finufft,
    which can vary significantly depending on tuning parameters like the number of threads.
    Order-of-magnitude speedups or slowdowns are possible. Unfortunately, it's very difficult
    to predict what will work best for a given problem on a given platform, so some experimentation
    may be necessary. For nthreads, start with 1 and increase until performance stops improving.
    Some other tuning parameters are listed under "finufft_kwargs" below.

    See the finufft documentation for the full list of tuning parameters:
    https://finufft.readthedocs.io/en/latest/opts.html


    Parameters
    ----------
    t : array-like
        The time values, shape (N_t,)
    y : array-like
        The data values, shape (N_t,) or (N_y, N_t)
    dy : array-like
        The uncertainties of the data values, broadcastable to `y`
    fmin : float
        The minimum frequency of the periodogram.
    df : float
        The frequency bin width.
    Nf : int
        The number of frequency bins.
    nthreads : int, optional
        The number of threads to use. The default behavior is to use (N_t / 4) * (Nf / 2^15) threads,
        capped to the number of available CPUs. This is a heuristic that may not work well in all cases.
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

    # TODO: better upsampfrac heuristics?
    default_finufft_kwargs = dict(
        eps='default',
        upsampfac=1.25,
        fftw=FFTW_ESTIMATE,
        debug=int(verbose),
    )

    finufft_kwargs = {**default_finufft_kwargs, **(finufft_kwargs or {})}

    dtype = t.dtype

    if finufft_kwargs['eps'] == 'default':
        if dtype == np.float32:
            finufft_kwargs['eps'] = 1e-5
        else:
            finufft_kwargs['eps'] = 1e-9
    if 'backend' in finufft_kwargs:
        raise ValueError('backend should not be passed as a keyword argument')

    cdtype = np.complex128 if dtype == np.float64 else np.complex64

    # treat 1D arrays as a batch of size 1
    squeeze_output = y.ndim == 1
    y = np.atleast_2d(y)
    dy = np.atleast_2d(dy)

    # If fit_mean, we need to transform (t,yw) and (t,w),
    # so we stack yw and w into a single array to allow a batched transform.
    # Regardless, we need to do a separate (t2,w2) transform.
    Nbatch, N = y.shape

    if nthreads is None:
        nthreads = max(1, Nbatch // 4) * max(1, Nf // (1 << 15))
        nthreads = min(nthreads, MAX_THREADS)

    if verbose:
        print(
            f'nifty-ls finufft: Using {nthreads} {"thread" if nthreads == 1 else "threads"}'
        )

    # could probably be more than finufft nthreads in many cases,
    # but it's conceptually cleaner to keep them the same, and it
    # will almost never matter in practice
    nthreads_helpers = nthreads

    if fit_mean:
        yw_w_shape = (2 * Nbatch, N)
    else:
        yw_w_shape = (Nbatch, N)

    yw_w = np.empty(yw_w_shape, dtype=cdtype)
    w2 = np.empty(dy.shape, dtype=cdtype)

    yw = yw_w[:Nbatch]
    w = yw_w[Nbatch:]

    t_helpers = -timer()
    if not _no_cpp_helpers:
        t1 = np.empty_like(t)
        t2 = np.empty_like(t)

        norm = np.empty(Nbatch, dtype=dtype)

        cpu_helpers.process_finufft_inputs(
            t1,  # output
            t2,  # output
            yw,  # output
            w,  # output
            w2,  # output
            norm,  # output
            t,  # input
            y,  # input
            dy,  # input
            fmin,
            df,
            Nf,
            center_data,
            fit_mean,
            normalization.lower() == 'psd',
            nthreads_helpers,
        )
    else:
        t1 = 2 * np.pi * df * t
        t2 = 2 * t1

        t1 = t1.astype(dtype, copy=False)
        t2 = t2.astype(dtype, copy=False)
        y = y.astype(dtype, copy=False)
        dy = dy.astype(dtype, copy=False)

        w2[:] = dy**-2.0
        w2.real /= w2.real.sum(axis=-1, keepdims=True)

        if center_data or fit_mean:
            y = y - (w2.real * y).sum(axis=-1, keepdims=True)

        norm = (w2.real * y**2).sum(axis=-1, keepdims=True)

        Nshift = Nf // 2

        phase_shift1 = np.exp(1j * (Nshift + fmin / df) * t1)
        phase_shift2 = np.exp(1j * (Nshift + fmin / df) * t2)

        t1 %= 2 * np.pi
        t2 %= 2 * np.pi

        yw[:] = y * w2.real
        if fit_mean:
            # Up to now, w and w2 are identical
            # but now they pick up a different phase shift
            w[:] = w2

        yw_w *= phase_shift1
        w2 *= phase_shift2

    t_helpers += timer()

    t_finufft = -timer()
    plan_solo = finufft.Plan(
        nufft_type=1,
        n_modes_or_dim=(Nf,),
        n_trans=Nbatch,
        dtype=cdtype,
        nthreads=nthreads,
        **finufft_kwargs,
    )

    if fit_mean:
        # fit_mean needs two transforms with the same NU points,
        # so we pack them into one transform
        plan_pair = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=(Nf,),
            n_trans=2 * Nbatch,
            dtype=cdtype,
            nthreads=nthreads,
            **finufft_kwargs,
        )

        # S, C = trig_sum(t, w, **kwargs)
        # tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
        plan_pair.setpts(t1)
        f1_fw = plan_pair.execute(yw_w)

    else:
        plan_solo.setpts(t1)
        f1_fw = plan_solo.execute(yw_w)

    f1 = f1_fw[:Nbatch]
    fw = f1_fw[Nbatch:]

    plan_solo.setpts(t2)
    f2 = plan_solo.execute(w2)

    t_finufft += timer()

    t_helpers -= timer()
    if not _no_cpp_helpers:
        norm_enum = dict(
            standard=cpu_helpers.NormKind.Standard,
            model=cpu_helpers.NormKind.Model,
            log=cpu_helpers.NormKind.Log,
            psd=cpu_helpers.NormKind.PSD,
        )[normalization.lower()]

        power = np.empty(f1.shape, dtype=dtype)
        cpu_helpers.process_finufft_outputs(
            power,
            f1,
            fw,
            f2,
            norm,
            norm_enum,
            fit_mean,
            nthreads_helpers,
        )
    else:
        if fit_mean:
            tan_2omega_tau = (f2.imag - 2 * fw.imag * fw.real) / (
                f2.real - (fw.real * fw.real - fw.imag * fw.imag)
            )
        else:
            tan_2omega_tau = f2.imag / f2.real
        S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
        C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
        Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
        Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)

        YC = f1.real * Cw + f1.imag * Sw
        YS = f1.imag * Cw - f1.real * Sw
        CC = 0.5 * (1 + f2.real * C2w + f2.imag * S2w)
        SS = 0.5 * (1 - f2.real * C2w - f2.imag * S2w)

        if fit_mean:
            CC -= (fw.real * Cw + fw.imag * Sw) ** 2
            SS -= (fw.imag * Cw - fw.real * Sw) ** 2

        power = YC * YC / CC + YS * YS / SS

        if normalization == 'standard':
            power /= norm
        elif normalization == 'model':
            power /= norm - power
        elif normalization == 'log':
            power = -np.log(1 - power / norm)
        elif normalization == 'psd':
            power *= 0.5 * (dy**-2.0).sum()
        else:
            raise ValueError(f'Unknown normalization: {normalization}')

    t_helpers += timer()

    if verbose:
        print(
            f'nifty-ls finufft: FINUFFT took {t_finufft:.4g} sec, pre-/post-processing took {t_helpers:.4g} sec'
        )

    if squeeze_output:
        power = power.squeeze()
    return power
