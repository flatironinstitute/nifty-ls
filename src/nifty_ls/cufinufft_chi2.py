"""
This module provides a CUDA-accelerated implementation of the Lomb-Scargle fast chi-squared
periodogram via cufinufft and cupy.
"""

from __future__ import annotations

from timeit import default_timer as timer
from itertools import chain

from .utils import same_dtype_or_raise

try:
    import cupy as cp
    import cufinufft
except ImportError as e:
    raise ImportError(
        'cufinufft and cupy are required for this module. Did you install with "pip install nifty-ls[cuda]"?'
    ) from e


if not cp.is_available():
    raise ImportError('CUDA is not available, cannot use cufinufft backend.')

__all__ = ['lombscargle']


def lombscargle(
    t,
    y,
    fmin,
    df,
    Nf,
    dy=None,
    center_data=True,
    fit_mean=True,
    normalization='standard',
    copy_result_to_host=True,
    verbose=False,
    cufinufft_kwargs=None,
    nterms=1,
):
    """
    Compute the Lomb-Scargle periodogram (fastchi2 method) using the cufinufft backend.

    Performance Tuning
    ------------------
    cufinufft is not as finicky to tune as finufft. The default parameters are probably
    fine for most cases, but you may want to experiment with the `eps` and `gpu_method`.

    The cufinufft documentation has more information on performance tuning:

    https://finufft.readthedocs.io/en/latest/c_gpu.html#algorithm-performance-options

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
    center_data : bool, optional
        Whether to center the data before computing the periodogram. Default is True.
    fit_mean : bool, optional
        Whether to fit a mean value to the data before computing the periodogram. Default is True.
    normalization : str, optional
        The normalization method to use. One of ['standard', 'model', 'log', 'psd']. Default is 'standard'.
    copy_result_to_host : bool, optional
        If True, the result will be copied to host (CPU) memory before returning.
        This is usually desired unless you plan to do further computation on the GPU.
    verbose : bool, optional
        Whether to print diagnostic messages.
    cufinufft_kwargs : dict, optional
        Additional keyword arguments to pass to the `cufinufft.Plan()` constructor.
        Particular cufinufft parameters of interest are:
        - `eps`: the requested precision [1e-9 for double precision and 1e-5 for single precision]
        - `gpu_method`: the method to use on the GPU [1]
    nterms : int, optional
        Number of Fourier terms in the fit
    """

    if nterms == 0 and not fit_mean:
        raise ValueError('Cannot have nterms = 0 without fitting bias')

    default_cufinufft_kwargs = dict(eps='default', gpu_method=1)

    cufinufft_kwargs = {**default_cufinufft_kwargs, **(cufinufft_kwargs or {})}

    same_dtype_or_raise(t=t, y=y, dy=dy)

    dtype = t.dtype

    if cufinufft_kwargs['eps'] == 'default':
        if dtype == cp.float32:
            cufinufft_kwargs['eps'] = 1e-5
        else:
            cufinufft_kwargs['eps'] = 1e-9

    if 'backend' in cufinufft_kwargs:
        raise ValueError('backend should not be passed as a keyword argument')

    cdtype = cp.complex128 if dtype == cp.float64 else cp.complex64

    if dy is None:
        dy = dtype.type(1.0)

    t_copy = -timer()

    # transfer arrays to GPU if not already there
    t = cp.asarray(t)
    y = cp.asarray(y)
    dy = cp.asarray(dy)

    t_copy += timer()

    t_prepost = -timer()

    # treat 1D arrays as a batch of size 1
    squeeze_output = y.ndim == 1
    y = cp.atleast_2d(y)

    # If fit_mean, we need to transform (t,yw) and (t,w),
    # so we stack yw and w into a single array to allow a batched transform.
    # Regardless, we need to do a separate (t2,w2) transform.
    Nbatch, N = y.shape

    # Broadcast dy to match the shape of t and y
    dy_broadcasted = cp.broadcast_to(dy, (Nbatch, N))

    # Force fit them into 2 arrays to use the batched finufft transform for yw and w
    yw_w_shape = (2 * Nbatch, N)
    yw_w = cp.empty(yw_w_shape, dtype=cdtype)

    yw = yw_w[:Nbatch]
    w = yw_w[Nbatch:]

    # begin preprocessing
    # TODO: actually pre-/post-processing is a significant fraction of the time for small transforms
    # Pre‐allocate memory for trig matrix in dtype, since it not store complex numbers
    # 2*nterms + 1 terms for w, nterms + 1 terms for yw. Fetching the Plan
    nSW = 2 * nterms + 1
    nSY = nterms + 1
    Sw = cp.empty((Nbatch, nSW, Nf), dtype=dtype)  # Shape(Nbatch, nSW, Nf) and initlize
    Cw = cp.empty((Nbatch, nSW, Nf), dtype=dtype)
    Syw = cp.empty((Nbatch, nSY, Nf), dtype=dtype)
    Cyw = cp.empty((Nbatch, nSY, Nf), dtype=dtype)

    t1 = 2 * cp.pi * df * t
    t1 = t1.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    dy = dy.astype(dtype, copy=False)

    # w_base equivalent to w2 in fastfinufft
    w_base = (dy_broadcasted**-2.0).astype(dtype)
    w2s = cp.sum(w_base.real, axis=-1, keepdims=True)  # sum over N, shape: (Nbatch, 1)

    if center_data or fit_mean:
        y = y - ((w_base * y).sum(axis=-1, keepdims=True) / w2s)
        yws = cp.zeros((Nbatch, 1), dtype=dtype)  # shape: (Nbatch, 1)
    else:
        yws = (y * w_base).sum(
            axis=-1, keepdims=True
        )  # shape: (Nbatch, 1) to match w2s

    norm = (w_base * y**2).sum(axis=-1, keepdims=True)

    yw[:] = y * w_base
    w[:] = w_base

    # SCw = [(cp.zeros(Nf), ws * cp.ones(Nf))]
    # SCyw = [(cp.zeros(Nf), yws * cp.ones(Nf))]
    Sw[:, 0, :] = 0
    Cw[:, 0, :] = w2s  # broadcasting w2s from (Nbatch, 1) to (Nbatch, Nf)
    Syw[:, 0, :] = 0
    Cyw[:, 0, :] = yws  # broadcasting yws from (Nbatch, 1) to (Nbatch, Nf)

    # This function applies a time shift to the reference time t1 and computes
    # the corresponding phase shifts. It then creates a new batch of weights
    # by multiplying the input weights with the phase shifts.
    def compute_t(time_shift, yw_w):
        tn = time_shift * t1
        tn = tn.astype(dtype, copy=False)
        phase_shiftn = cp.exp(1j * ((Nf // 2) + fmin / df) * tn)  # shape = (N,)

        # Build a brand-new "batch" of phased weights for this i:
        yw_w_s = (yw_w * phase_shiftn).astype(cdtype)  # broadcasting explicit
        return tn, yw_w_s

    t_prepost += timer()

    t_cufinufft = -timer()

    # Loop over harmonics from i = 0 to nterms (inclusive)
    # For each frequency term pass yw_w as input:
    #   - Weighted data: y_i × w  at time points t
    #   - Pure weights: w  at time points t
    # Both share the same time coordinates (t), so we can batch them together
    # Use a single NUFFT plan to transform both y_i × w and w simultaneously and efficiently.
    plan_yw = cufinufft.Plan(
        nufft_type=1,
        n_modes=(Nf,),
        n_trans=2 * Nbatch,  # paired processing of y * w and w
        dtype=cdtype,
        **cufinufft_kwargs,
    )
    for j in range(1, nterms + 1):
        tj, yw_w_j = compute_t(j, yw_w)
        plan_yw.setpts(tj)
        f1_fw = plan_yw.execute(yw_w_j)
        # TODO: use out parameter in finufft.Plan.execute() to
        # write directly to Sw/Cw/Syw/Cyw arrays and avoid copying
        Sw[:, j, :] = f1_fw[Nbatch:].imag  # yw
        Cw[:, j, :] = f1_fw[Nbatch:].real
        Syw[:, j, :] = f1_fw[:Nbatch].imag
        Cyw[:, j, :] = f1_fw[:Nbatch].real

    # Since in fastchi2, the freq_factor of w includes terms
    # from 1 to 2*nterms, we need one more loop to handle
    # the result of the transform for indices nterms + 1 to 2*nterms(inclusive).
    plan_w = cufinufft.Plan(
        nufft_type=1,
        n_modes=(Nf,),
        n_trans=Nbatch,
        dtype=cdtype,
        **cufinufft_kwargs,
    )
    for i in range(nterms + 1, 2 * nterms + 1):
        ti, yw_w_i = compute_t(i, yw_w)
        plan_w.setpts(ti)
        f2_all = plan_w.execute(yw_w_i[Nbatch:])  # w only
        Sw[:, i, :] = f2_all.imag
        Cw[:, i, :] = f2_all.real
    t_cufinufft += timer()

    # begin postprocessing
    t_prepost -= timer()

    order = [('C', 0)] if fit_mean else []
    order.extend(chain(*([('S', i), ('C', i)] for i in range(1, nterms + 1))))
    # Build-up matrices at each frequency
    power = cp.zeros((Nbatch, Nf), dtype=dtype)

    nterms_order = len(order)
    # Pre-allocate matrices, used for all batches and frequencies
    XTX = cp.empty((Nbatch, nterms_order, nterms_order, Nf), dtype=dtype)
    XTy = cp.empty((Nbatch, nterms_order, Nf), dtype=dtype)

    # Build XTX matrix
    for row_idx, (B_code, B_m) in enumerate(order):
        for col_idx, (A_code, A_m) in enumerate(order):
            # Determine the operation type based on code combination
            operation_type = A_code + B_code

            if operation_type == 'CC':
                # eg. CC: 0.5 * (Cw[abs(m - n)] + Cw[m + n])
                XTX[:, row_idx, col_idx, :] = 0.5 * (
                    Cw[:, abs(A_m - B_m), :] + Cw[:, A_m + B_m, :]
                )
            elif operation_type == 'SS':
                XTX[:, row_idx, col_idx, :] = 0.5 * (
                    Cw[:, abs(A_m - B_m), :] - Cw[:, A_m + B_m, :]
                )
            elif operation_type == 'CS':
                XTX[:, row_idx, col_idx, :] = 0.5 * (
                    cp.sign(B_m - A_m) * Sw[:, abs(B_m - A_m), :] + Sw[:, B_m + A_m, :]
                )
            elif operation_type == 'SC':
                XTX[:, row_idx, col_idx, :] = 0.5 * (
                    cp.sign(A_m - B_m) * Sw[:, abs(A_m - B_m), :] + Sw[:, A_m + B_m, :]
                )
            else:
                raise ValueError(f'Unknown operation type: {operation_type}')

    # Build XTy vector
    for idx, (code, m) in enumerate(order):
        if code == 'S':
            XTy[:, idx, :] = Syw[:, m, :]
        elif code == 'C':
            XTy[:, idx, :] = Cyw[:, m, :]
        else:
            raise ValueError(f'Unknown code: {code}')

    # Solve the linear system for all batches and frequencies at once
    # XTX shape: (Nbatch, nterms_order, nterms_order, Nf)
    # XTy shape: (Nbatch, nterms_order, Nf)
    # We need to solve XTX[b,:,:,f] @ solution[b,:,f] = XTy[b,:,f] for all b,f

    XTX_trans = XTX.transpose(
        0, 3, 1, 2
    )  # Shape: (Nbatch, Nf, nterms_order, nterms_order)
    XTy_trans = XTy.transpose(0, 2, 1)  # Shape: (Nbatch, Nf, nterms_order)

    # Solve linear systems
    solutions = cp.linalg.solve(
        XTX_trans, XTy_trans
    )  # Shape: (Nbatch, Nf, nterms_order)
    raw_power = cp.sum(solutions * XTy_trans, axis=2)  # Shape: (Nbatch, Nf)

    # Apply normalization to all batches at once
    if normalization == 'standard':
        power = raw_power / norm
    elif normalization == 'model':
        power = raw_power / (norm - raw_power)
    elif normalization == 'log':
        power = -cp.log(1 - raw_power / norm)
    elif normalization == 'psd':
        power = raw_power * 0.5
    else:
        raise ValueError(f'Unknown normalization: {normalization}')

    # treat 1D arrays as a batch of size 1
    if squeeze_output:
        power = power.squeeze()

    t_prepost += timer()

    if copy_result_to_host:
        t_copy -= timer()
        power = power.get()
        t_copy += timer()

    if verbose:
        print(f'[nifty-ls cufinufft] HtoD + DtoH = {t_copy:.4g} sec')
        print(f'[nifty-ls cufinufft] pre/post = {t_prepost:.4g} sec')
        print(f'[nifty-ls cufinufft] cufinufft = {t_cufinufft:.4g} sec')

    return power
