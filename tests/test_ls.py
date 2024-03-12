import numpy as np
import pytest


def gen_data(N=100, nobj=None, seed=5043, dtype=np.float64):
    rng = np.random.default_rng(seed)

    t = np.sort(rng.random(N, dtype=dtype)) * 123
    freqs = rng.random((nobj, 1) if nobj else 1, dtype=dtype) * 10 + 1
    y = np.sin(freqs * t) + 1.23
    dy = rng.random(y.shape, dtype=dtype) * 0.1 + 0.01

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


@pytest.fixture(scope='module')
def batched_data():
    return gen_data(nobj=100)


@pytest.fixture(scope='module')
def batched_bench_data():
    return gen_data(N=3_000, nobj=100)


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
    normalization='standard',
):
    from astropy.timeseries import LombScargle

    ls = LombScargle(
        t,
        y,
        dy,
        fit_mean=fit_mean,
        center_data=center_data,
        normalization=normalization,
    )

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

    assert np.allclose(nifty_res, brute_res, rtol=1e-9 if dtype == np.float64 else 1e-5)


def test_batched(batched_data, Nf=1000):
    """Check various batching modes"""
    import nifty_ls

    t = batched_data['t']
    y_batch = batched_data['y']
    dy_batch = batched_data['dy']

    fmin = 0.1
    fmax = 10.0

    nifty_res = nifty_ls.lombscargle(t, y_batch, dy_batch, fmin=fmin, fmax=fmax, Nf=Nf)

    brute_res = np.empty((len(y_batch), Nf), dtype=y_batch.dtype)
    for i in range(len(y_batch)):
        brute_res[i] = astropy(
            t, y_batch[i], dy_batch[i], fmin, fmax, Nf, use_fft=False
        )

    dtype = t.dtype

    assert np.allclose(nifty_res, brute_res, rtol=1e-9 if dtype == np.float64 else 1e-5)


def test_normalization(data, Nf=1000):
    """Check that the normalization modes work as expected"""
    import nifty_ls

    fmin = 0.1
    fmax = 10.0

    for norm in ['standard', 'model', 'log', 'psd']:
        nifty_res = nifty_ls.lombscargle(
            **data,
            fmin=fmin,
            fmax=fmax,
            Nf=Nf,
            normalization=norm,
        )
        astropy_res = astropy(
            **data,
            fmin=fmin,
            fmax=fmax,
            Nf=Nf,
            use_fft=False,
            normalization=norm,
        )
        assert np.allclose(nifty_res, astropy_res)


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


@pytest.mark.parametrize('Nf', [1_000])
def test_no_cpp_helpers(data, batched_data, Nf):
    """Check that the _no_cpp_helpers flag works as expected for batched and unbatched"""
    import nifty_ls

    fmin = 0.1
    fmax = 10.0

    nifty_power = nifty_ls.lombscargle(
        **data, fmin=fmin, fmax=fmax, Nf=Nf, backend_kwargs=dict(_no_cpp_helpers=False)
    )

    nocpp_power = nifty_ls.lombscargle(
        **data, fmin=fmin, fmax=fmax, Nf=Nf, backend_kwargs=dict(_no_cpp_helpers=True)
    )

    assert np.allclose(nifty_power, nocpp_power)

    nifty_power_batched = nifty_ls.lombscargle(
        **batched_data,
        fmin=fmin,
        fmax=fmax,
        Nf=Nf,
        backend_kwargs=dict(_no_cpp_helpers=False),
    )

    nocpp_power_batched = nifty_ls.lombscargle(
        **batched_data,
        fmin=fmin,
        fmax=fmax,
        Nf=Nf,
        backend_kwargs=dict(_no_cpp_helpers=True),
    )

    assert np.allclose(nifty_power_batched, nocpp_power_batched)


# TODO: cuda test
# TODO: center_data, fit_mean, normalization tests


@pytest.mark.parametrize('Nf', [10_000, 100_000, 1_000_000])
class TestPerf:
    """Benchmark nifty-ls versus astropy's FFT-based implementation.

    This pytest-benchmark integration is experimental; the plugin doesn't
    seem very well maintained, so we'll have to see how reliable/useful
    this integration is.
    """

    fmin = 0.1
    fmax = 10.0

    def test_nifty(self, bench_data, Nf, benchmark):
        import nifty_ls

        benchmark(
            nifty_ls.lombscargle, **bench_data, fmin=self.fmin, fmax=self.fmax, Nf=Nf
        )

    def test_astropy(self, bench_data, Nf, benchmark):
        benchmark(
            astropy, **bench_data, fmin=self.fmin, fmax=self.fmax, Nf=Nf, use_fft=True
        )

    # Usually this benchmark isn't very useful, since one will always use the
    # compiled extensions in practice, but if looking at the performance
    # of the extensions themselves, it might be interesting.
    # def test_nifty_nohelpers(self, bench_data, Nf, benchmark):
    #     import nifty_ls

    #     benchmark(nifty_ls.lombscargle, **bench_data, fmin=0.1, fmax=10.0, Nf=Nf,
    #               backend_kwargs=dict(_no_cpp_helpers=True))


@pytest.mark.parametrize('Nf', [1_000])
class TestBatchedPerf:
    fmin = 0.1
    fmax = 10.0

    def test_batched(self, batched_bench_data, Nf, benchmark):
        import nifty_ls

        benchmark(
            nifty_ls.lombscargle,
            **batched_bench_data,
            fmin=self.fmin,
            fmax=self.fmax,
            Nf=Nf,
        )

    def test_unbatched(self, batched_bench_data, Nf, benchmark):
        import nifty_ls

        t = batched_bench_data['t']
        y_batch = batched_bench_data['y']
        dy_batch = batched_bench_data['dy']

        def _nifty():
            for i in range(len(y_batch)):
                nifty_ls.lombscargle(
                    t, y_batch[i], dy_batch[i], fmin=self.fmin, fmax=self.fmax, Nf=Nf
                )

        benchmark(_nifty)

    def test_astropy_unbatched(self, batched_bench_data, Nf, benchmark):
        t = batched_bench_data['t']
        y_batch = batched_bench_data['y']
        dy_batch = batched_bench_data['dy']

        def _astropy():
            for i in range(len(y_batch)):
                astropy(
                    t,
                    y_batch[i],
                    dy_batch[i],
                    fmin=self.fmin,
                    fmax=self.fmax,
                    Nf=Nf,
                    use_fft=True,
                )

        benchmark(_astropy)
