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

For development (with automatic rebuilds enabled by default in `pyproject.toml`):
```console
$ pip install nanobind scikit-build-core
$ pip install -e .[test] --no-build-isolation
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

In many cases, accelerating your periodogram is as simple as setting the method
in your Astropy Lomb Scargle code! More advanced usage, such as computing multiple
periodograms in parallel, should go directly through the nifty-ls interface.

You can also tune nifty-ls by passing arguments via the `method_kws` dict:

```python
frequency, power = LombScargle(t, y, method="fastnifty").autopower()
```

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
============================================================== test session starts ==============================================================
platform linux -- Python 3.10.13, pytest-8.0.2, pluggy-1.4.0
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=True min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /mnt/home/lgarrison/nifty-ls
configfile: pyproject.toml
plugins: benchmark-4.0.0, asdf-2.15.0, anyio-3.6.2, hypothesis-6.23.1
collected 15 items                                                                                                                              

tests/test_ls.py ...............                                                                                                          [100%]


------------------------------------------------------------------------------------ benchmark '1000': 3 tests -------------------------------------------------------------------------------------
Name (time in ms)                     Min                 Max                Mean            StdDev              Median               IQR            Outliers      OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_batched[1000]                97.2214 (1.0)      103.6985 (1.0)       99.2986 (1.0)      1.8384 (1.05)      98.7446 (1.0)      1.6039 (1.17)          2;1  10.0706 (1.0)          10           1
test_unbatched[1000]             138.7306 (1.43)     143.6479 (1.39)     139.9287 (1.41)     1.7432 (1.0)      139.3874 (1.41)     1.3677 (1.0)           1;1   7.1465 (0.71)          7           1
test_astropy_unbatched[1000]     336.1734 (3.46)     341.5773 (3.29)     338.9706 (3.41)     1.9510 (1.12)     338.8120 (3.43)     2.1159 (1.55)          2;0   2.9501 (0.29)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------- benchmark '10000': 2 tests ----------------------------------------------------------------------------
Name (time in ms)          Min               Max              Mean            StdDev            Median               IQR            Outliers       OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_nifty[10000]       1.9664 (1.0)      5.7140 (1.0)      2.0224 (1.0)      0.2512 (1.0)      1.9839 (1.0)      0.0144 (1.0)          9;29  494.4615 (1.0)         397           1
test_astropy[10000]     5.8273 (2.96)     7.5100 (1.31)     6.0420 (2.99)     0.3477 (1.38)     5.8986 (2.97)     0.1854 (12.90)        9;10  165.5077 (0.33)         90           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------- benchmark '100000': 2 tests ------------------------------------------------------------------------------
Name (time in ms)            Min                Max               Mean            StdDev             Median               IQR            Outliers      OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_nifty[100000]        9.8257 (1.0)      15.4502 (1.0)      10.0691 (1.0)      0.7908 (1.0)       9.9051 (1.0)      0.0232 (1.0)          4;15  99.3140 (1.0)          56           1
test_astropy[100000]     43.8031 (4.46)     53.5885 (3.47)     45.5756 (4.53)     2.7517 (3.48)     44.5811 (4.50)     2.1073 (90.84)         2;2  21.9416 (0.22)         19           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------ benchmark '1000000': 2 tests ------------------------------------------------------------------------------------
Name (time in ms)                Min                   Max                  Mean             StdDev                Median                IQR            Outliers     OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_nifty[1000000]          94.1771 (1.0)        161.5810 (1.0)        112.0059 (1.0)      23.7536 (8.40)       100.3770 (1.0)      20.8909 (4.81)          1;1  8.9281 (1.0)           7           1
test_astropy[1000000]     1,060.3990 (11.26)    1,067.4917 (6.61)     1,063.4248 (9.49)      2.8276 (1.0)      1,063.2405 (10.59)     4.3464 (1.0)           2;0  0.9404 (0.11)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
============================================================== 15 passed in 19.96s ==============================================================
```

## Developer Notes
Early development of the package was done in the https://github.com/dfm/nufft-ls repo.
