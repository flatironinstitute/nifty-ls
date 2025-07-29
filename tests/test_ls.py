"""
Tests for the nifty-ls Lomb-Scargle implementation, including comparison against
Astropy's implementation.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import nifty_ls
import nifty_ls.backends
import nifty_ls.utils
from nifty_ls.test_helpers.utils import gen_data, astropy_ls, astropy_ls_fastchi2


def rtol(dtype, Nf):
    """Use a relative tolerance that accounts for the condition number of the problem"""
    if dtype == np.float32:
        # NB we don't test float32
        return max(1e-3, 1e-7 * Nf)
    elif dtype == np.float64:
        return max(1e-5, 1e-9 * Nf)
    else:
        raise ValueError(f'Unknown dtype {dtype}')


@pytest.fixture(scope='module')
def data():
    return gen_data()


@pytest.fixture(scope='module')
def batched_data():
    return gen_data(Nbatch=100)


@pytest.fixture(scope='module')
def nifty_backend(request):
    avail_backends = nifty_ls.core.AVAILABLE_BACKENDS

    if request.param in avail_backends:
        fn = partial(nifty_ls.lombscargle, backend=request.param)
        return fn, request.param
    else:
        pytest.skip(f'Backend {request.param} is not available')


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [('auto', None), ('auto', 1), ('auto', 2)],
    indirect=['nifty_backend'],
)
def test_auto_backend_selection(data, nifty_backend, nterms, Nf=1000):
    """Test that the auto backend selection works as expected"""
    backend_fn, _ = nifty_backend
    nifty_res = backend_fn(**data, Nf=Nf, nterms=nterms).power

    if nterms and nterms > 1:
        brute_res = astropy_ls_fastchi2(**data, nterms=nterms, Nf=Nf, use_fft=False)
    else:
        brute_res = astropy_ls(**data, Nf=Nf, use_fft=False)
    dtype = data['t'].dtype

    np.testing.assert_allclose(
        nifty_res,
        brute_res,
        rtol=rtol(dtype, Nf),
    )


def test_backend_error_handling(data, Nf=1000):
    """Test error handling for incompatible backend and nterms combinations"""
    # Test error when using finufft with nterms > 1
    with pytest.raises(
        ValueError,
        match='Backend "finufft" only supports nterms == 1. Use "cufinufft_chi2" or "finufft_chi2" for nterms > 1.',
    ):
        nifty_ls.lombscargle(**data, Nf=Nf, backend='finufft', nterms=2)

    # Test error with unknown backend
    with pytest.raises(ValueError, match='Unknown or unavailable backend'):
        nifty_ls.lombscargle(**data, Nf=Nf, backend='non_existed_backend', nterms=1)


@pytest.mark.parametrize('Nf', [1_000, 10_000, 100_000])
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft', None),
        ('finufft', 1),
        ('finufft_chi2', 1),
        ('finufft_chi2', 2),
        ('cufinufft', None),
        ('cufinufft_chi2', 1),
        ('cufinufft_chi2', 2),
    ],
    indirect=['nifty_backend'],
)
def test_lombscargle(data, Nf, nifty_backend, nterms):
    """Check that the basic implementation agrees with the brute-force Astropy answer"""

    backend_fn, backend_name = nifty_backend
    nifty_res = backend_fn(**data, Nf=Nf, nterms=nterms).power
    if backend_name == 'cufinufft_chi2' or backend_name == 'finufft_chi2':
        brute_res = astropy_ls_fastchi2(**data, nterms=nterms, Nf=Nf, use_fft=False)
    else:
        brute_res = astropy_ls(**data, Nf=Nf, use_fft=False)
    dtype = data['t'].dtype

    np.testing.assert_allclose(
        nifty_res,
        brute_res,
        rtol=rtol(dtype, Nf),
    )


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft', None),
        ('finufft', 1),
        ('finufft_chi2', 1),
        ('finufft_chi2', 2),
        ('cufinufft', None),
        ('cufinufft_chi2', 1),
        ('cufinufft_chi2', 2),
    ],
    indirect=['nifty_backend'],
)
def test_batched(batched_data, nifty_backend, nterms, Nf=1000):
    """Check various batching modes with different backends and nterms"""

    backend_fn, backend_name = nifty_backend
    nifty_res = backend_fn(**batched_data, Nf=Nf, nterms=nterms).power

    t = batched_data['t']
    y_batch = batched_data['y']
    dy_batch = batched_data['dy']
    fmin = batched_data['fmin']
    fmax = batched_data['fmax']

    brute_res = np.empty((len(y_batch), Nf), dtype=y_batch.dtype)
    for i in range(len(y_batch)):
        if backend_name == 'finufft_chi2' or backend_name == 'cufinufft_chi2':
            brute_res[i] = astropy_ls_fastchi2(
                t, y_batch[i], dy_batch[i], fmin, fmax, Nf, nterms=nterms, use_fft=False
            )
        else:
            brute_res[i] = astropy_ls(
                t, y_batch[i], dy_batch[i], fmin, fmax, Nf, use_fft=False
            )

    dtype = t.dtype

    np.testing.assert_allclose(
        nifty_res,
        brute_res,
        # the batched case runs 100 examples
        # and is more likely to probe the tails of the error distribution
        rtol=rtol(dtype, Nf) * 10,
    )


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft', None),
        ('finufft_chi2', 2),
        ('cufinufft', None),
        ('cufinufft_chi2', 2),
    ],
    indirect=['nifty_backend'],
)
def test_normalization(data, nifty_backend, nterms, Nf=1000):
    """Check that the normalization modes work as expected"""

    backend_fn, backend_name = nifty_backend

    for norm in ['standard', 'model', 'log', 'psd']:
        nifty_res = backend_fn(
            **data,
            Nf=Nf,
            nterms=nterms,
            normalization=norm,
        ).power
        if backend_name == 'finufft_chi2' or backend_name == 'cufinufft_chi2':
            astropy_res = astropy_ls_fastchi2(
                **data,
                Nf=Nf,
                nterms=nterms,
                use_fft=False,
                normalization=norm,
            )
        else:
            astropy_res = astropy_ls(
                **data,
                Nf=Nf,
                use_fft=False,
                normalization=norm,
            )
        dtype = data['t'].dtype
        np.testing.assert_allclose(nifty_res, astropy_res, rtol=rtol(dtype, Nf))


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [('finufft', 1), ('finufft_chi2', 2), ('cufinufft', 1), ('cufinufft_chi2', 2)],
    indirect=['nifty_backend'],
)
def test_astropy_hook(data, nifty_backend, nterms, Nf=1000):
    from astropy.timeseries import LombScargle

    ls = LombScargle(data['t'], data['y'], data['dy'], fit_mean=True, center_data=True)
    ls_chi2 = LombScargle(
        data['t'], data['y'], data['dy'], fit_mean=True, center_data=True, nterms=nterms
    )

    freq = np.linspace(data['fmin'], data['fmax'], Nf, endpoint=True)

    backend_fn, backend_name = nifty_backend
    nifty_power = backend_fn(
        **data, Nf=Nf, fit_mean=True, center_data=True, nterms=nterms
    ).power
    if backend_name == 'cufinufft_chi2' or backend_name == 'finufft_chi2':
        astropy_power = ls_chi2.power(
            freq,
            method='fastnifty_chi2',
            assume_regular_frequency=True,
            method_kwds=dict(backend=backend_name),
        )
    else:
        astropy_power = ls.power(
            freq,
            method='fastnifty',
            assume_regular_frequency=True,
            method_kwds=dict(backend=backend_name),
        )

    # same backend, ought to match very closely
    np.testing.assert_allclose(astropy_power, nifty_power)


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [('finufft', None), ('finufft_chi2', 1), ('finufft_chi2', 2)],
    indirect=['nifty_backend'],
)
def test_no_cpp_helpers(data, batched_data, nifty_backend, nterms, Nf=1000):
    """Check that the _no_cpp_helpers flag works as expected for batched and unbatched"""

    backend_fn, _ = nifty_backend

    # Unbatched case
    power_cpp_unbatched = backend_fn(
        **data, Nf=Nf, nterms=nterms, _no_cpp_helpers=False
    ).power
    power_nocpp_unbatched = backend_fn(
        **data, Nf=Nf, nterms=nterms, _no_cpp_helpers=True
    ).power
    np.testing.assert_allclose(power_cpp_unbatched, power_nocpp_unbatched)

    # Batched case
    power_cpp_batched = backend_fn(
        **batched_data, Nf=Nf, nterms=nterms, _no_cpp_helpers=False
    ).power
    power_nocpp_batched = backend_fn(
        **batched_data, Nf=Nf, nterms=nterms, _no_cpp_helpers=True
    ).power
    np.testing.assert_allclose(power_cpp_batched, power_nocpp_batched)

    # Batched case without dy
    batched_data = batched_data.copy()
    batched_data['dy'] = None
    power_cpp = backend_fn(
        **batched_data, Nf=Nf, nterms=nterms, _no_cpp_helpers=False
    ).power
    power_nocpp = backend_fn(
        **batched_data, Nf=Nf, nterms=nterms, _no_cpp_helpers=True
    ).power
    np.testing.assert_allclose(power_cpp, power_nocpp)


@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft', None),
        ('finufft_chi2', 2),
        ('cufinufft', None),
        ('cufinufft_chi2', 2),
    ],
    indirect=['nifty_backend'],
)
def test_center_data(data, center_data, nterms, nifty_backend, Nf=1000):
    backend_fn, backend_name = nifty_backend

    center_nifty = backend_fn(
        **data, Nf=Nf, nterms=nterms, center_data=center_data
    ).power

    if backend_name == 'finufft_chi2' or backend_name == 'cufinufft_chi2':
        center_astropy = astropy_ls_fastchi2(
            **data,
            Nf=Nf,
            nterms=nterms,
            use_fft=False,
            center_data=center_data,
        )
    else:
        center_astropy = astropy_ls(
            **data,
            Nf=Nf,
            use_fft=False,
            center_data=center_data,
        )
    dtype = data['t'].dtype

    np.testing.assert_allclose(center_nifty, center_astropy, rtol=rtol(dtype, Nf))


@pytest.mark.parametrize('fit_mean', [True, False])
@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft', None),
        ('finufft_chi2', 2),
        ('cufinufft', None),
        ('cufinufft_chi2', 2),
    ],
    indirect=['nifty_backend'],
)
def test_fit_mean(data, fit_mean, nifty_backend, nterms, Nf=1000):
    backend_fn, backend_name = nifty_backend

    fitmean_nifty = backend_fn(**data, Nf=Nf, fit_mean=fit_mean, nterms=nterms).power

    if backend_name == 'finufft_chi2' or backend_name == 'cufinufft_chi2':
        fitmean_astropy = astropy_ls_fastchi2(
            **data,
            Nf=Nf,
            nterms=nterms,
            fit_mean=fit_mean,
            use_fft=False,
        )
    else:
        fitmean_astropy = astropy_ls(
            **data,
            Nf=Nf,
            fit_mean=fit_mean,
            use_fft=False,
        )

    dtype = data['t'].dtype

    np.testing.assert_allclose(fitmean_nifty, fitmean_astropy, rtol=rtol(dtype, Nf))


@pytest.mark.parametrize(
    'nifty_backend,nterms',
    [
        ('finufft', None),
        ('finufft_chi2', 2),
        ('cufinufft', None),
        ('cufinufft_chi2', 2),
    ],
    indirect=['nifty_backend'],
)
def test_dy_none(data, batched_data, nifty_backend, nterms, Nf=1000):
    """Test that `dy = None` works properly"""

    backend_fn, backend_name = nifty_backend

    # Unbatched case
    data = data.copy()
    data['dy'] = None
    nifty_res = backend_fn(**data, Nf=Nf, nterms=nterms).power

    if backend_name == 'finufft_chi2' or backend_name == 'cufinufft_chi2':
        astropy_res = astropy_ls_fastchi2(**data, Nf=Nf, nterms=nterms, use_fft=False)
    else:
        astropy_res = astropy_ls(**data, Nf=Nf, use_fft=False)

    dtype = data['t'].dtype

    np.testing.assert_allclose(nifty_res, astropy_res, rtol=rtol(dtype, Nf))

    # Batched case
    # the dy = None case involves broadcasting; better test batched mode too
    batched_data = batched_data.copy()
    batched_data['dy'] = None

    nifty_res = backend_fn(**batched_data, Nf=Nf, nterms=nterms).power

    astropy_res = np.empty((len(batched_data['y']), Nf), dtype=batched_data['y'].dtype)
    for i in range(len(batched_data['y'])):
        if backend_name == 'finufft_chi2' or backend_name == 'cufinufft_chi2':
            astropy_res[i] = astropy_ls_fastchi2(
                batched_data['t'],
                batched_data['y'][i],
                None,
                batched_data['fmin'],
                batched_data['fmax'],
                Nf,
                nterms=nterms,
                use_fft=False,
            )
        else:
            astropy_res[i] = astropy_ls(
                batched_data['t'],
                batched_data['y'][i],
                None,
                batched_data['fmin'],
                batched_data['fmax'],
                Nf,
                use_fft=False,
            )

    dtype = batched_data['t'].dtype

    np.testing.assert_allclose(nifty_res, astropy_res, rtol=rtol(dtype, Nf))


def test_backends(data, nterms=1, Nf=1000):
    """Test that all the backends give the same answer,
    without reference to astropy
    """

    backends = nifty_ls.core.AVAILABLE_BACKENDS
    if len(backends) < 2:
        pytest.skip('Need more than one backend to compare')

    powers = {
        backend: nifty_ls.lombscargle(
            **data, Nf=Nf, nterms=nterms, backend=backend
        ).power
        for backend in backends
    }

    dtype = data['t'].dtype
    for backend1, power1 in powers.items():
        for backend2, power2 in powers.items():
            if backend1 == backend2:
                continue
            np.testing.assert_allclose(power1, power2, rtol=rtol(dtype, Nf))


# GH #58
def test_mixed_dtypes(data):
    """Test that calling lombscargle with mixed dtypes raises an exception."""
    backends = nifty_ls.core.AVAILABLE_BACKENDS

    data_mixed = data.copy()
    data_mixed['t'] = data_mixed['t'].astype(np.float32)
    data_mixed['y'] = data_mixed['y'].astype(np.float64)
    data_mixed['dy'] = data_mixed['dy'].astype(np.float64)

    for backend in backends:
        with pytest.raises(ValueError, match='dtype'):
            nifty_ls.lombscargle(**data_mixed, backend=backend)

    data_mixed = data.copy()
    data_mixed['t'] = data_mixed['t'].astype(np.float32)
    data_mixed['y'] = data_mixed['y'].astype(np.float32)
    data_mixed['dy'] = data_mixed['dy'].astype(np.float64)

    for backend in backends:
        with pytest.raises(ValueError, match='dtype'):
            nifty_ls.lombscargle(**data_mixed, backend=backend)
