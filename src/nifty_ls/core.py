import os

import finufft
import numpy as np

from . import cpu

DEFAULT_NTHREAD = len(os.sched_getaffinity(0))

FFTW_MEASURE = 0
FFTW_ESTIMATE = 64


def lombscargle(
    t,
    y,
    dy,
    fmin=None,
    fmax=None,
    Nf=None,
    nthreads=None,
    normalization='standard',
    backend='finufft',
    backend_kwargs=None,
):
    """
    Parameters
    ----------
    t : array_like or list of array_like
        Times of observations. If `t` is an array, then it must be one dimensional (shape (N,)).
        If `t` is a list of arrays, then each of the arrays must be one dimensional,
        but they can have different lengths (with `t[i]` having shape `(N_i,)`).
    y : array_like or list of array_like
        Measurement values. If `y` is an array, then it must have shape `([M,] N)`, where `M` is the number of
        light curves sharing the same observation times and `N` is the number of observations.
        If `y` is a list of arrays, then each of the arrays must be one dimensional,
        but they can have different lengths (`y[i]` must have shape `(N_i,)`).
    """

    if Nf is None:
        pass  # TODO

    if fmax is None:
        pass  # TODO

    if fmin is None:
        pass  # TODO

    df = (fmax - fmin) / (Nf - 1)  # fmax inclusive

    if backend == 'finufft':
        return lombscargle_finufft(
            t,
            y,
            dy,
            fmin,
            df,
            Nf,
            nthreads,
            normalization,
            **(backend_kwargs or {}),
        )
    else:
        raise ValueError(f'Unknown backend: {backend}. Available backends are: finufft')


def lombscargle_finufft(
    t,
    y,
    dy,
    fmin,
    df,
    Nf,
    nthreads=None,
    normalization='standard',
    _no_cpp_helpers=False,
    **finufft_kwargs,
):
    """
    Computes `nthreads` finuffts in parallel, or fewer if newer NU points are given.
    The rest of the threads will be passed down into finufft.
    """

    # TODO: better upsampfrac heuristics?
    default_finufft_kwargs = dict(eps='default', upsampfac=1.25, fftw=FFTW_ESTIMATE)

    finufft_kwargs = {**default_finufft_kwargs, **finufft_kwargs}

    if nthreads is None:
        # TODO: better thread heuristics?
        nthreads = 1 if Nf <= 10**5 else DEFAULT_NTHREAD

    if nthreads < 1:
        nthreads = DEFAULT_NTHREAD

    dtype = t.dtype

    if finufft_kwargs['eps'] == 'default':
        if dtype == np.float32:
            finufft_kwargs['eps'] = 1e-5
        else:
            finufft_kwargs['eps'] = 1e-9

    cdtype = np.complex128 if dtype == np.float64 else np.complex64

    # treat 1D arrays as a batch of size 1
    squeeze_output = y.ndim == 1
    y = np.atleast_2d(y)
    dy = np.atleast_2d(dy)

    if not _no_cpp_helpers:
        t1 = np.empty_like(t)
        t2 = np.empty_like(t)
        yw = np.empty(y.shape, dtype=cdtype)
        w = np.empty(dy.shape, dtype=cdtype)
        norm = np.empty(len(y), dtype=dtype)

        cpu.process_finufft_inputs(
            t1,  # output
            t2,  # output
            yw,  # output
            w,  # output
            norm,  # output
            t,  # input
            y,  # input
            dy,  # input
            fmin,
            df,
            Nf,
            normalization.lower() == 'psd',
        )
    else:
        t1 = 2 * np.pi * df * t
        t2 = 2 * t1

        t1 = t1.astype(dtype, copy=False)
        t2 = t2.astype(dtype, copy=False)
        y = y.astype(dtype, copy=False)
        dy = dy.astype(dtype, copy=False)

        w = dy**-2.0
        w /= w.sum(axis=-1, keepdims=True)

        norm = (w * y**2).sum(axis=-1, keepdims=True)

        Nshift = Nf // 2

        phase_shift1 = np.exp(1j * (Nshift + fmin / df) * t1)
        phase_shift2 = np.exp(1j * (Nshift + fmin / df) * t2)

        t1 %= 2 * np.pi
        t2 %= 2 * np.pi

        yw = y * w

        yw = yw * phase_shift1
        w = w * phase_shift2

    plan = finufft.Plan(
        nufft_type=1,
        n_modes_or_dim=(Nf,),
        n_trans=len(yw),
        dtype=cdtype,
        nthreads=nthreads,
        **finufft_kwargs,
    )
    plan.setpts(t1)
    f1 = plan.execute(yw)

    plan.setpts(t2)
    f2 = plan.execute(w)

    if not _no_cpp_helpers:
        norm_enum = dict(
            standard=cpu.NormKind.Standard,
            model=cpu.NormKind.Model,
            log=cpu.NormKind.Log,
            psd=cpu.NormKind.PSD,
        )[normalization.lower()]

        power = np.empty(f1.shape, dtype=dtype)
        cpu.process_finufft_outputs(power, f1, f2, norm, norm_enum)
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

    if squeeze_output:
        power = power.squeeze()
    return power
