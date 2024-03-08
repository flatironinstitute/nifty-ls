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
First, clone the repo:
```console
$ git clone https://www.github.com/flatironinstitute/nifty-ls
```

To install with CPU support:

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

Kind (1) will be very fast, as finufft has native support for computing multiple
simultaneous transforms with the same non-uniform points (observation times).

Kind (2) uses multi-threading to compute multiple distinct transforms in parallel.


## Performance


## Accuracy


## Testing
First, install from source (`pip install .[test]`). Then, from the repo root, run:

```console
$ pytest
```

The tests are defined in the `tests/` directory, and include a mini-benchmark of
nifty-ls and Astropy:

<div style='font-family: 'Droid Sans Mono', 'monospace', monospace, monospace; font-size: 14px;'><div><span>$ pytest                                                                                                                                                                                                                                          </span></div><div><span></span><span style='font-weight: bold;'>============================================================= test session starts =============================================================</span><span>                                                                                                                              </span></div><div><span>platform linux -- Python 3.10.13, pytest-8.0.2, pluggy-1.4.0                                                                                                                                                                                                                 </span></div><div><span>benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=True min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)                                                                                              </span></div><div><span>rootdir: /mnt/home/lgarrison/nifty-ls                                                                                                                                                                                                                                        </span></div><div><span>configfile: pyproject.toml                                                                                                                                                                                                                                                   </span></div><div><span>plugins: benchmark-4.0.0, asdf-2.15.0, anyio-3.6.2, hypothesis-6.23.1                                                                                                                                                                                                        </span></div><div><span></span><span style='font-weight: bold;'>collected 10 items                                                                                                                            </span><span>                                                                                                                               </span></div><div><span>                                                                                                                                                                                                                                                                             </span></div><div><span>tests/test_ls.py </span><span style='color: #0dbc79;'>..........                                                                                                             [100%]</span><span>                                                                                                                               </span></div><div><span>                                                                                                                                                                                                                                                                             </span></div><div><span>                                                                                                                                                                                                                                                                             </span></div><div><span></span><span style='color: #e5e510;'>---------------------------------------------------------------------------- benchmark '10000': 2 tests ----------------------------------------------------------------------------</span><span>                                                                                         </span></div><div><span>Name (time in ms)          Min               Max              Mean            StdDev            Median               IQR            Outliers       OPS            Rounds  Iterations                                                                                         </span></div><div><span></span><span style='color: #e5e510;'>------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</span><span>                                                                                         </span></div><div><span>test_nifty[10000]     </span><span style='color: #0dbc79; font-weight: bold;'>  1.9492 (1.0)      4.7619 (1.0)      2.0031 (1.0)      0.1980 (1.0)      1.9812 (1.0)      0.0147 (1.0)    </span><span>      6;15</span><span style='color: #0dbc79; font-weight: bold;'>  499.2362 (1.0)    </span><span>     392           1                                                                                         </span></div><div><span>test_astropy[10000]   </span><span style='color: #cd3131; font-weight: bold;'>  5.9338 (3.04)     7.6690 (1.61)     6.2057 (3.10)     0.3699 (1.87)     6.0530 (3.06)     0.3010 (20.45)  </span><span>      14;9</span><span style='color: #cd3131; font-weight: bold;'>  161.1425 (0.32)   </span><span>      93           1                                                                                         </span></div><div><span></span><span style='color: #e5e510;'>------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</span><span>                                                                                         </span></div><div><span>                                                                                                                                                                                                                                                                             </span></div><div><span></span><span style='color: #e5e510;'>----------------------------------------------------------------------------- benchmark '100000': 2 tests ------------------------------------------------------------------------------</span><span>                                                                                     </span></div><div><span>Name (time in ms)            Min                Max               Mean            StdDev             Median               IQR            Outliers      OPS            Rounds  Iterations                                                                                     </span></div><div><span></span><span style='color: #e5e510;'>----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</span><span>                                                                                     </span></div><div><span>test_nifty[100000]     </span><span style='color: #0dbc79; font-weight: bold;'>   9.9320 (1.0)      19.8285 (1.0)      10.3198 (1.0)      1.2494 (1.0)      10.0192 (1.0)      0.0372 (1.0)    </span><span>      3;14</span><span style='color: #0dbc79; font-weight: bold;'>  96.9015 (1.0)    </span><span>      86           1                                                                                     </span></div><div><span>test_astropy[100000]   </span><span style='color: #cd3131; font-weight: bold;'>  44.2575 (4.46)     51.3133 (2.59)     45.5472 (4.41)     1.9427 (1.55)     44.6738 (4.46)     1.5047 (40.42)  </span><span>       2;2</span><span style='color: #cd3131; font-weight: bold;'>  21.9553 (0.23)   </span><span>      19           1                                                                                     </span></div><div><span></span><span style='color: #e5e510;'>----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</span><span>                                                                                     </span></div><div><span>                                                                                                                                                                                                                                                                             </span></div><div><span></span><span style='color: #e5e510;'>------------------------------------------------------------------------------------ benchmark '1000000': 2 tests ------------------------------------------------------------------------------------</span><span>                                                                       </span></div><div><span>Name (time in ms)                Min                   Max                  Mean             StdDev                Median                IQR            Outliers     OPS            Rounds  Iterations                                                                       </span></div><div><span></span><span style='color: #e5e510;'>------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</span><span>                                                                       </span></div><div><span>test_nifty[1000000]     </span><span style='color: #0dbc79; font-weight: bold;'>     95.7697 (1.0)        111.8821 (1.0)        102.9107 (1.0)       6.0323 (1.0)        101.0534 (1.0)       9.4641 (1.0)    </span><span>       3;0</span><span style='color: #0dbc79; font-weight: bold;'>  9.7172 (1.0)    </span><span>       7           1                                                                       </span></div><div><span>test_astropy[1000000]   </span><span style='color: #cd3131; font-weight: bold;'>  1,273.6085 (13.30)    1,306.3966 (11.68)    1,285.4819 (12.49)    13.2602 (2.20)     1,278.8661 (12.66)    17.5778 (1.86)   </span><span>       1;0</span><span style='color: #cd3131; font-weight: bold;'>  0.7779 (0.08)   </span><span>       5           1                                                                       </span></div><div><span></span><span style='color: #e5e510;'>------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</span><span>                                                                       </span></div><div><span>                                                                                                                                                                                                                                                                             </span></div><div><span>Legend:                                                                                                                                                                                                                                                                      </span></div><div><span>  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.                                                                                                                                                                </span></div><div><span>  OPS: Operations Per Second, computed as 1 / Mean                                                                                                                                                                                                                           </span></div><div><span></span><span style='color: #0dbc79;'>============================================================= </span><span style='color: #0dbc79; font-weight: bold;'>10 passed</span><span style='color: #0dbc79;'> in 16.39s =============================================================</span></div></div>

## Developer Notes
Early development of the package was done in the https://github.com/dfm/nufft-ls repo.
