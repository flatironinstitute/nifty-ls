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
from nifty_ls.test_helpers.utils import gen_data, astropy_ls


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
        return partial(nifty_ls.lombscargle, backend=request.param)
    else:
        pytest.skip(f'Backend {request.param} is not available')


@pytest.mark.parametrize('Nf', [1_000, 10_000, 100_000])
@pytest.mark.parametrize('nifty_backend', ['finufft', 'cufinufft'], indirect=True)
def test_lombscargle(data, Nf, nifty_backend):
    """Check that the basic implementation agrees with the brute-force Astropy answer"""

    nifty_res = nifty_backend(**data, Nf=Nf)['power']
    brute_res = astropy_ls(**data, Nf=Nf, use_fft=False)

    dtype = data['t'].dtype

    np.testing.assert_allclose(
        nifty_res,
        brute_res,
        rtol=1e-6 if dtype == np.float64 else 1e-3,
    )


@pytest.mark.parametrize('nifty_backend', ['finufft', 'cufinufft'], indirect=True)
def test_batched(batched_data, nifty_backend, Nf=1000):
    """Check various batching modes"""

    nifty_res = nifty_backend(**batched_data, Nf=Nf)['power']

    t = batched_data['t']
    y_batch = batched_data['y']
    dy_batch = batched_data['dy']
    fmin = batched_data['fmin']
    fmax = batched_data['fmax']

    brute_res = np.empty((len(y_batch), Nf), dtype=y_batch.dtype)
    for i in range(len(y_batch)):
        brute_res[i] = astropy_ls(
            t, y_batch[i], dy_batch[i], fmin, fmax, Nf, use_fft=False
        )

    dtype = t.dtype

    np.testing.assert_allclose(
        nifty_res, brute_res, rtol=1e-6 if dtype == np.float64 else 1e-3
    )


@pytest.mark.parametrize('nifty_backend', ['finufft', 'cufinufft'], indirect=True)
def test_normalization(data, nifty_backend, Nf=1000):
    """Check that the normalization modes work as expected"""

    for norm in ['standard', 'model', 'log', 'psd']:
        nifty_res = nifty_backend(
            **data,
            Nf=Nf,
            normalization=norm,
        )['power']
        astropy_res = astropy_ls(
            **data,
            Nf=Nf,
            use_fft=False,
            normalization=norm,
        )
        dtype = data['t'].dtype
        np.testing.assert_allclose(
            nifty_res, astropy_res, rtol=1e-6 if dtype == np.float64 else 1e-3
        )


@pytest.mark.parametrize('backend', nifty_ls.backends.BACKEND_NAMES)
def test_astropy_hook(data, backend, Nf=1000):
    """Check that the fastnifty method is available in astropy's Lomb Scargle"""
    if backend not in nifty_ls.core.AVAILABLE_BACKENDS:
        pytest.skip(f'Backend {backend} is not available')

    from astropy.timeseries import LombScargle

    ls = LombScargle(data['t'], data['y'], data['dy'], fit_mean=True, center_data=True)

    freq = np.linspace(0.1, 10.0, Nf, endpoint=True)

    astropy_power = ls.power(
        freq,
        method='fastnifty',
        assume_regular_frequency=True,
        method_kwds=dict(backend=backend),
    )

    nifty_power = nifty_ls.lombscargle(
        **data, Nf=Nf, fit_mean=True, center_data=True, backend=backend
    )['power']

    # same backend, ought to match very closely
    np.testing.assert_allclose(astropy_power, nifty_power)


def test_no_cpp_helpers(data, batched_data, Nf=1000):
    """Check that the _no_cpp_helpers flag works as expected for batched and unbatched"""

    nifty_power = nifty_ls.lombscargle(**data, Nf=Nf, _no_cpp_helpers=False)['power']

    nocpp_power = nifty_ls.lombscargle(**data, Nf=Nf, _no_cpp_helpers=True)['power']

    np.testing.assert_allclose(nifty_power, nocpp_power)

    nifty_power_batched = nifty_ls.lombscargle(
        **batched_data,
        Nf=Nf,
        _no_cpp_helpers=False,
    )['power']

    nocpp_power_batched = nifty_ls.lombscargle(
        **batched_data,
        Nf=Nf,
        _no_cpp_helpers=True,
    )['power']

    np.testing.assert_allclose(nifty_power_batched, nocpp_power_batched)


@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize('nifty_backend', ['finufft', 'cufinufft'], indirect=True)
def test_center_data(data, center_data, nifty_backend, Nf=1000):
    center_nifty = nifty_backend(**data, Nf=Nf, center_data=center_data)['power']

    center_astropy = astropy_ls(**data, Nf=Nf, center_data=center_data, use_fft=False)

    dtype = data['t'].dtype

    np.testing.assert_allclose(
        center_nifty, center_astropy, rtol=1e-6 if dtype == np.float64 else 1e-3
    )


@pytest.mark.parametrize('fit_mean', [True, False])
@pytest.mark.parametrize('nifty_backend', ['finufft', 'cufinufft'], indirect=True)
def test_fit_mean(data, fit_mean, nifty_backend, Nf=1000):
    fitmean_nifty = nifty_backend(**data, Nf=Nf, fit_mean=fit_mean)['power']

    fitmean_astropy = astropy_ls(
        **data,
        Nf=Nf,
        fit_mean=fit_mean,
        use_fft=False,
    )

    dtype = data['t'].dtype

    np.testing.assert_allclose(
        fitmean_nifty, fitmean_astropy, rtol=1e-6 if dtype == np.float64 else 1e-3
    )


def test_backends(data, Nf=1000):
    """Test that all the backends give the same answer"""

    backends = nifty_ls.core.AVAILABLE_BACKENDS
    if len(backends) < 2:
        pytest.skip('Need more than one backend to compare')

    powers = {
        backend: nifty_ls.lombscargle(**data, Nf=Nf, backend=backend)['power']
        for backend in backends
    }

    for backend1, power1 in powers.items():
        for backend2, power2 in powers.items():
            if backend1 == backend2:
                continue
            np.testing.assert_allclose(power1, power2, rtol=1e-6)
