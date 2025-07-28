"""
Performance tests for nifty-ls, comparing various backends and astropy's implementation.

The benchmarking is done using the `pytest-benchmark` plugin, which is working well for
the moment but doesn't seem to be very actively maintained, so we may want to keep an
eye out for alternatives.
"""

from __future__ import annotations

import pytest

import nifty_ls
import nifty_ls.backends
from nifty_ls.test_helpers.utils import gen_data, astropy_ls, astropy_ls_fastchi2


@pytest.fixture(scope='module')
def batched_bench_data():
    return gen_data(N=3_000, Nbatch=100)


@pytest.fixture(scope='module')
def bench_data():
    return gen_data(N=3_000)


@pytest.fixture(params=nifty_ls.backends.STANDARD_BACKEND_NAMES + ['astropy'])
def standard_backend(request):
    """Parametrize over all nifty-ls backends, and astropy."""
    if request.param not in nifty_ls.core.AVAILABLE_BACKENDS and request.param not in (
        'astropy',
        'astropy_fastchi2',
    ):
        pytest.skip(f'Backend {request.param} is not available')
    return request.param


@pytest.fixture(params=nifty_ls.backends.CHI2_BACKEND_NAMES + ['astropy_fastchi2'])
def chi2_backend(request):
    """Parametrize over all nifty-ls backends, and astropy."""
    if request.param not in nifty_ls.core.AVAILABLE_BACKENDS and request.param not in (
        'astropy',
        'astropy_fastchi2',
    ):
        pytest.skip(f'Backend {request.param} is not available')
    return request.param


class TestPerf:
    """Benchmark nifty-ls versus astropy's FFT-based implementation."""

    @pytest.mark.parametrize('Nf', [10_000, 100_000, 1000_000])
    def test_standard(self, bench_data, Nf, benchmark, standard_backend):
        benchmark.group = 'standard'
        if standard_backend == 'astropy':
            benchmark(astropy_ls, **bench_data, Nf=Nf, use_fft=True)
        else:
            benchmark(
                nifty_ls.lombscargle, **bench_data, Nf=Nf, backend=standard_backend
            )
        # Usually this benchmark isn't very useful, since one will always use the
        # compiled extensions in practice, but if looking at the performance
        # of the extensions themselves, it might be interesting.
        # def test_nifty_nohelpers(self, bench_data, Nf, benchmark):
        #     import nifty_ls

        #     benchmark(nifty_ls.lombscargle, **bench_data, fmin=0.1, fmax=10.0, Nf=Nf,
        #               _no_cpp_helpers=True).power

    @pytest.mark.parametrize('Nf', [10_000])
    def test_chi2_nterms4(self, bench_data, Nf, benchmark, chi2_backend):
        """Benchmark chi2 backends with nterms=4 and fixed Nf=10000."""
        benchmark.group = 'chi2_nterms4'
        if chi2_backend == 'astropy_fastchi2':
            benchmark(astropy_ls_fastchi2, **bench_data, Nf=Nf, nterms=4)
        else:
            benchmark(
                nifty_ls.lombscargle,
                **bench_data,
                Nf=Nf,
                backend=chi2_backend,
                nterms=4,
            )


@pytest.mark.parametrize('Nf', [1_000])
class TestBatchedPerf:
    @pytest.mark.parametrize(
        'standard_backend',
        list(
            set(nifty_ls.backends.STANDARD_BACKEND_NAMES)
            & set(nifty_ls.core.AVAILABLE_BACKENDS)
        ),
        indirect=True,
    )
    def test_batched_standard(
        self, batched_bench_data, Nf, benchmark, standard_backend
    ):
        benchmark.group = 'batched_standard'
        benchmark(
            nifty_ls.lombscargle,
            **batched_bench_data,
            Nf=Nf,
            backend=standard_backend,
        )

    @pytest.mark.parametrize(
        'standard_backend',
        list(
            set(nifty_ls.backends.STANDARD_BACKEND_NAMES)
            & set(nifty_ls.core.AVAILABLE_BACKENDS)
        )
        + ['astropy'],
        indirect=True,
    )
    def test_unbatched_standard(
        self, batched_bench_data, Nf, benchmark, standard_backend
    ):
        benchmark.group = 'batched_standard'
        t = batched_bench_data['t']
        y_batch = batched_bench_data['y']
        dy_batch = batched_bench_data['dy']
        fmin = batched_bench_data['fmin']
        fmax = batched_bench_data['fmax']

        def _nifty():
            for i in range(len(y_batch)):
                nifty_ls.lombscargle(
                    t,
                    y_batch[i],
                    dy_batch[i],
                    fmin=fmin,
                    fmax=fmax,
                    Nf=Nf,
                    backend=standard_backend,
                )

        def _astropy():
            for i in range(len(y_batch)):
                astropy_ls(
                    t,
                    y_batch[i],
                    dy_batch[i],
                    fmin=fmin,
                    fmax=fmax,
                    Nf=Nf,
                    use_fft=True,
                )

        if standard_backend == 'astropy':
            benchmark(_astropy)
        else:
            benchmark(_nifty)
