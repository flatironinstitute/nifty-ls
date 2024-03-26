"""
This module provides a CUDA-accelerated implementation of the Lomb-Scargle periodogram
via cufinufft and cupy.
"""

from __future__ import annotations

try:
    import cupy as cp
    import cufinufft
except ImportError as e:
    raise ImportError(
        'cufinufft and cupy are required for this module. Did you install with "pip install nifty-ls[cuda]"?'
    ) from e


if not cp.is_available():
    raise ImportError('CUDA is not available, cannot use cufinufft backend.')


def lombscargle(
    t,
    y,
    dy,
    fmin,
    df,
    Nf,
    center_data=True,
    fit_mean=True,
    normalization='standard',
    copy_result_to_cpu=True,
    **cufinufft_kwargs,
):
    """
    Computes `nthreads` finuffts in parallel, or fewer if newer NU points are given.
    The rest of the threads will be passed down into finufft.
    """

    # TODO: better upsampfrac heuristics?
    default_cufinufft_kwargs = dict(eps='default')

    cufinufft_kwargs = {**default_cufinufft_kwargs, **cufinufft_kwargs}

    dtype = t.dtype

    if cufinufft_kwargs['eps'] == 'default':
        if dtype == cp.float32:
            cufinufft_kwargs['eps'] = 1e-5
        else:
            cufinufft_kwargs['eps'] = 1e-9

    cdtype = cp.complex128 if dtype == cp.float64 else cp.complex64

    # transfer arrays to GPU if not already there
    t = cp.asarray(t)
    y = cp.asarray(y)
    dy = cp.asarray(dy)

    # treat 1D arrays as a batch of size 1
    squeeze_output = y.ndim == 1
    y = cp.atleast_2d(y)
    dy = cp.atleast_2d(dy)

    # If fit_mean, we need to transform (t,yw) and (t,w),
    # so we stack yw and w into a single array to allow a batched transform.
    # Regardless, we need to do a separate (t2,w2) transform.
    Nbatch, N = y.shape
    if fit_mean:
        yw_w_shape = (2 * Nbatch, N)
    else:
        yw_w_shape = (Nbatch, N)

    yw_w = cp.empty(yw_w_shape, dtype=cdtype)
    w2 = cp.empty(dy.shape, dtype=cdtype)

    yw = yw_w[:Nbatch]
    w = yw_w[Nbatch:]

    # begin preprocessing
    t1 = 2 * cp.pi * df * t
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

    phase_shift1 = cp.exp(1j * (Nshift + fmin / df) * t1)
    phase_shift2 = cp.exp(1j * (Nshift + fmin / df) * t2)

    t1 %= 2 * cp.pi
    t2 %= 2 * cp.pi

    yw[:] = y * w2.real
    if fit_mean:
        # Up to now, w and w2 are identical
        # but now they pick up a different phase shift
        w[:] = w2

    yw_w *= phase_shift1
    w2 *= phase_shift2

    plan_solo = cufinufft.Plan(
        nufft_type=1,
        n_modes=(Nf,),
        n_trans=Nbatch,
        dtype=cdtype,
        **cufinufft_kwargs,
    )

    if fit_mean:
        # fit_mean needs two transforms with the same NU points,
        # so we pack them into one transform
        plan_pair = cufinufft.Plan(
            nufft_type=1,
            n_modes=(Nf,),
            n_trans=2 * Nbatch,
            dtype=cdtype,
            **cufinufft_kwargs,
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

    # begin postprocessing
    if fit_mean:
        tan_2omega_tau = (f2.imag - 2 * fw.imag * fw.real) / (
            f2.real - (fw.real * fw.real - fw.imag * fw.imag)
        )
    else:
        tan_2omega_tau = f2.imag / f2.real
    S2w = tan_2omega_tau / cp.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    C2w = 1 / cp.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    Cw = cp.sqrt(0.5) * cp.sqrt(1 + C2w)
    Sw = cp.sqrt(0.5) * cp.sign(S2w) * cp.sqrt(1 - C2w)

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
        power = -cp.log(1 - power / norm)
    elif normalization == 'psd':
        power *= 0.5 * (dy**-2.0).sum()
    else:
        raise ValueError(f'Unknown normalization: {normalization}')

    if squeeze_output:
        power = power.squeeze()

    if copy_result_to_cpu:
        power = power.get()

    return power
