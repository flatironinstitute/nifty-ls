import concurrent.futures
import os
import sys

import numpy as np
import pytest

import nifty_ls

NIFTY_LS_FORCE_FREETHREADED_TEST = (
    os.environ.get('NIFTY_LS_FORCE_FREETHREADED_TEST', '0') == '1'
)


def nogil_or_exit():
    try:
        gil_enabled = sys._is_gil_enabled()
    except AttributeError:
        gil_enabled = True

    if gil_enabled:
        if NIFTY_LS_FORCE_FREETHREADED_TEST:
            pytest.fail(
                'NIFTY_LS_FORCE_FREETHREADED_TEST is set, but Python is not running in free-threaded mode.'
            )
        else:
            pytest.skip('Python is not running in free-threaded mode')


# NB we don't use pytest.mark.skipif here because we want to check the GIL status at runtime


def test_threaded_pool_finufft(N_periodograms=200, N_points=10000):
    """
    Test concurrent computation of periodograms using ThreadPoolExecutor with finufft backend.
    """

    nogil_or_exit()

    rng = np.random.default_rng(42)

    t = np.sort(rng.uniform(0, 100, size=N_points))
    frequencies = rng.uniform(0.1, 2.0, size=N_periodograms)

    # Generate different y values for each periodogram
    y_values = np.sin(2 * np.pi * frequencies[:, None] * t) + 0.1 * rng.normal(
        size=(N_periodograms, N_points)
    )

    # Parameters for lombscargle
    common_kwargs = {
        'backend': 'finufft',
        'nthreads': 1,  # not required, but recommended
    }

    def compute_periodogram(y):
        return nifty_ls.lombscargle(t, y, **common_kwargs)

    # Sequential computation
    sequential_results = nifty_ls.lombscargle(t, y_values, **common_kwargs).power

    # Parallel computation using ThreadPoolExecutor
    parallel_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_periodogram, y) for y in y_values]

    parallel_results = [future.result() for future in futures]

    # Make sure we have all results
    assert len(sequential_results) == N_periodograms
    assert len(parallel_results) == N_periodograms

    # Verify results are the same
    for seq_res, par_res in zip(sequential_results, parallel_results):
        np.testing.assert_allclose(seq_res, par_res.power, rtol=1e-4)
