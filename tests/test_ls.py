import numpy as np
import pytest


def gen_data(N=100, seed=5043, dtype=np.float64):
    rng = np.random.default_rng(seed)

    t = np.sort(rng.random(N, dtype=dtype)) * 123
    y = np.sin(20 * t)
    dy = rng.random(N, dtype=dtype) * 0.1 + 0.01

    t.setflags(write=False)
    y.setflags(write=False)
    dy.setflags(write=False)

    return dict(t=t, y=y, dy=dy)


@pytest.fixture(scope='module')
def data():
    return gen_data()


@pytest.fixture(scope='module')
def bench_data():
    return gen_data(N=3_000)


def astropy(
    t,
    y,
    dy,
    fmin,
    fmax,
    Nf,
    fit_mean=False,
    center_data=False,
    use_fft=False,
):
    from astropy.timeseries import LombScargle

    ls = LombScargle(t, y, dy, fit_mean=fit_mean, center_data=center_data)

    freq = np.linspace(fmin, fmax, Nf, endpoint=True)
    power = ls.power(
        freq,
        method='fast',
        method_kwds=dict(use_fft=use_fft),
        assume_regular_frequency=True,
    )

    return power


@pytest.mark.parametrize('Nf', [1_000, 10_000, 100_000])
def test_lombscargle(data, Nf):
    """Check that the basic implementation agrees with the brute-force Astropy answer"""
    import nifty_ls

    t = data['t']
    y = data['y']
    dy = data['dy']

    fmin = 0.1
    fmax = 10.0

    nifty_res = nifty_ls.lombscargle(t, y, dy, fmin=fmin, fmax=fmax, Nf=Nf)
    brute_res = astropy(t, y, dy, fmin, fmax, Nf, use_fft=False)

    dtype = t.dtype

    # breakpoint()
    assert np.allclose(nifty_res, brute_res, rtol=1e-9 if dtype == np.float64 else 1e-5)


def test_astropy_hook(data, Nf=1000):
    """Check that the fastnifty method is available in astropy's Lomb Scargle"""
    from astropy.timeseries import LombScargle
    import nifty_ls

    fmin = 0.1
    fmax = 10.0

    ls = LombScargle(**data, fit_mean=False, center_data=False)

    freq = np.linspace(0.1, 10.0, Nf, endpoint=True)
    astropy_power = ls.power(freq, method='fastnifty', assume_regular_frequency=True)

    nifty_power = nifty_ls.lombscargle(**data, fmin=fmin, fmax=fmax, Nf=Nf)

    # same backend, ought to match very closely
    assert np.allclose(astropy_power, nifty_power)


# TODO: batch test
# TODO: cuda test
# TODO: center_data, fit_mean, normalization tests


@pytest.mark.parametrize('Nf', [10_000, 100_000, 1_000_000])
class TestPerf:
    """Benchmark nifty-ls versus astropy's FFT-based implementation.

    This pytest-benchmark integration is experimental; the plugin doesn't
    seem very well maintained, so we'll have to see how reliable/useful
    this integration is.
    """

    def test_nifty(self, bench_data, Nf, benchmark):
        import nifty_ls

        benchmark(nifty_ls.lombscargle, **bench_data, fmin=0.1, fmax=10.0, Nf=Nf)

    def test_astropy(self, bench_data, Nf, benchmark):
        benchmark(astropy, **bench_data, fmin=0.1, fmax=10.0, Nf=Nf, use_fft=True)

    # Usually this benchmark isn't very useful, since one will always use the
    # compiled extensions in practice, but if looking at the performance
    # of the extensions themselves, it might be interesting.
    # def test_nifty_nohelpers(self, bench_data, Nf, benchmark):
    #     import nifty_ls

    #     benchmark(nifty_ls.lombscargle, **bench_data, fmin=0.1, fmax=10.0, Nf=Nf,
    #               backend_kwargs=dict(no_cpp_helpers=True))
