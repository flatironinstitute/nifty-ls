# nifty-ls
A fast Lomb-Scargle periodogram. It's nifty, and uses a NUFFT!

[![PyPI](https://img.shields.io/pypi/v/nifty-ls)](https://pypi.org/project/nifty-ls/) [![Tests](https://github.com/flatironinstitute/nifty-ls/actions/workflows/tests.yml/badge.svg)](https://github.com/flatironinstitute/nifty-ls/actions/workflows/tests.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/flatironinstitute/nifty-ls/main.svg)](https://results.pre-commit.ci/latest/github/flatironinstitute/nifty-ls/main)

> [!WARNING]
> This project is in a pre-release stage and will likely undergo breaking changes during development. Some of the instructions in the README are aspirational.

## Overview
The Lomb-Scargle periodogram, used for identifying periodicity in irregularly-spaced
observations, is useful but computationally expensive. However, it can be
phrased mathematically as a pair of non-uniform FFTs (NUFFTs). This allows us to
leverage Flatiron Institute's [finufft](https://github.com/flatironinstitute/finufft/)
package, which is really fast! It also enables GPU (CUDA) support and is
several orders of magnitude more accurate than
[Astropy's Lomb Scargle](https://docs.astropy.org/en/stable/timeseries/lombscargle.html)
with default settings for many regions of parameter space.


## Installation
### From PyPI
For CPU support:

```console
$ pip install nifty-ls
```

For GPU (CUDA) support:

```console
$ pip install nifty-ls[cuda]
```

The default is to install with CUDA 12 support; one can use `nifty-ls[cuda11]` instead for CUDA 11 support (installs `cupy-cuda11x`).

### From source
First, clone the repo and `cd` to the repo root:
```console
$ git clone https://www.github.com/flatironinstitute/nifty-ls
$ cd nifty-ls
```

Then, to install with CPU support:

```console
$ pip install .
```

To install with GPU (CUDA) support:

```console
$ pip install .[cuda]
```

or `.[cuda11]` for CUDA 11.

For development (with automatic rebuilds enabled by default in `pyproject.toml`):
```console
$ pip install nanobind scikit-build-core
$ pip install -e .[test] --no-build-isolation
```

Developers may also be interested in setting these keys in `pyproject.toml`:

```toml
[tool.scikit-build]
cmake.build-type = "Debug"
cmake.verbose = true
install.strip = false
```

## Usage
### From Astropy
Importing `nifty_ls` makes nifty-ls available via `method="fastnifty"` in
Astropy's LombScargle module. The name is prefixed with "fast" as it's part
of the fast family of methods that assume a regularly-spaced frequency grid.

```python
import nifty_ls
from astropy.timeseries import LombScargle
frequency, power = LombScargle(t, y, method="fastnifty").autopower()
```

To use the CUDA (cufinufft) backend, pass the appropriate arguments via `method_kws`:

```python
frequency, power = LombScargle(t, y, method="fastnifty", method_kws=dict(backend="cufinufft")).autopower()
```


In many cases, accelerating your periodogram is as simple as setting the method
in your Astropy Lomb Scargle code! More advanced usage, such as computing multiple
periodograms in parallel, should go directly through the nifty-ls interface.


### From nifty-ls (native interface)
nifty-ls has its own interface that offers more flexibility than the Astropy
interface for batched periodograms.

A single periodogram can be computed as:

```python
import nifty_ls
# with automatic frequency grid:
nifty_res = nifty_ls.lombscargle(t, y, dy)

# with user-specified frequency grid:
nifty_res = nifty_ls.lombscargle(t, y, dy, fmin=0.1, fmax=10, Nf=10**6)
```

Two kinds of batching are supported:

1. multiple periodograms with the same observation times, and
2. multiple periodograms with distinct observation times.

Kind (1) uses finufft's native support for computing multiple simultaneous transforms
with the same non-uniform points (observation times).

Kind (2) uses multi-threading to compute multiple distinct transforms in parallel.

These two kinds of batching can be combined.

#### Batched Periodograms

```python
N_t = 100
N_obj = 10
Nf = 200

rng = np.random.default_rng()
t = rng.random(N_t)
freqs = rng.random(N_obj).reshape(-1,1)
y_batch = np.sin(freqs * t)
dy_batch = rng.random(y.shape)

batched_power = nifty_ls.lombscargle(t, y_batch, dy_batch, Nf=Nf)
print(batched_power.shape)  # (10, 200)
```

### Limitations
The code only supports frequency grids with fixed spacing; however, finufft does
support type 3 NUFFTs (non-uniform to non-uniform), which would enable arbitrary
frequency grids. It's not clear how useful this is, so it hasn't been implemented,
but please open a GitHub issue if this is of interest to you.

## Performance


## Accuracy


## Testing
First, install from source (`pip install .[test]`). Then, from the repo root, run:

```console
$ pytest
```

The tests are defined in the `tests/` directory, and include a mini-benchmark of
nifty-ls and Astropy:

```
$ pytest
=================================================== test session starts ===================================================
platform linux -- Python 3.10.13, pytest-8.0.2, pluggy-1.4.0
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=True min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /mnt/home/lgarrison/nifty-ls
configfile: pyproject.toml
plugins: benchmark-4.0.0, asdf-2.15.0, anyio-3.6.2, hypothesis-6.23.1
collected 36 items                                                                                                        

tests/test_ls.py ......................                                                                             [ 61%]
tests/test_perf.py ..............                                                                                   [100%]


----------------------------------------- benchmark 'Nf=1000': 5 tests ----------------------------------------
Name (time in ms)                       Min                Mean            StdDev            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------
test_batched[cufinufft-1000]        21.6041 (1.0)       23.6710 (1.0)      2.4283 (4.81)         32           1
test_batched[finufft-1000]         123.0226 (5.69)     123.4696 (5.22)     0.5053 (1.0)           8           1
test_unbatched[finufft-1000]       174.9419 (8.10)     175.4410 (7.41)     0.5325 (1.05)          6           1
test_unbatched[astropy-1000]       442.6370 (20.49)    443.1872 (18.72)    0.5587 (1.11)          5           1
test_unbatched[cufinufft-1000]     507.3561 (23.48)    517.5353 (21.86)    8.0330 (15.90)         5           1
---------------------------------------------------------------------------------------------------------------

--------------------------------- benchmark 'Nf=10000': 3 tests ----------------------------------
Name (time in ms)            Min              Mean            StdDev            Rounds  Iterations
--------------------------------------------------------------------------------------------------
test[finufft-10000]       2.5704 (1.0)      2.5933 (1.0)      0.0697 (1.0)         360           1
test[cufinufft-10000]     5.2686 (2.05)     5.3355 (2.06)     0.3733 (5.36)        154           1
test[astropy-10000]       7.5168 (2.92)     7.6362 (2.94)     0.1090 (1.56)        121           1
--------------------------------------------------------------------------------------------------

----------------------------------- benchmark 'Nf=100000': 3 tests ----------------------------------
Name (time in ms)              Min               Mean            StdDev            Rounds  Iterations
-----------------------------------------------------------------------------------------------------
test[cufinufft-100000]      6.0493 (1.0)       6.1367 (1.0)      0.4577 (26.49)       160           1
test[finufft-100000]       13.6574 (2.26)     13.6905 (2.23)     0.0173 (1.0)          71           1
test[astropy-100000]       48.1205 (7.95)     48.7020 (7.94)     0.4037 (23.36)        20           1
-----------------------------------------------------------------------------------------------------

------------------------------------- benchmark 'Nf=1000000': 3 tests --------------------------------------
Name (time in ms)                  Min                  Mean            StdDev            Rounds  Iterations
------------------------------------------------------------------------------------------------------------
test[cufinufft-1000000]         8.2061 (1.0)          8.6021 (1.0)      0.6529 (1.0)          97           1
test[finufft-1000000]          87.3530 (10.64)       90.1813 (10.48)    1.8716 (2.87)         10           1
test[astropy-1000000]       1,399.2794 (170.52)   1,402.8130 (163.08)   5.0011 (7.66)          5           1
------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
=================================================== 36 passed in 32.20s ===================================================
```

## Authors
nifty-ls was originally implemented by [Lehman Garrison](https://github.com/lgarrison)
based on work done by [Dan Foreman-Mackey](https://github.com/dfm) in the
[dfm/nufft-ls](https://github.com/dfm/nufft-ls) repo, with consulting from
[Alex Barnett](https://github.com/ahbarnett).

nifty-ls builds directly on top of the excellent finufft package by Alex Barnett
and others (see the [finufft Acknowledgements](https://finufft.readthedocs.io/en/latest/ackn.html)).
