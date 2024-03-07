# nifty-ls
A fast Lomb-Scargle periodogram. It's nifty, and uses a NUFFT!

> [!WARNING]
> This project is in a pre-release stage and will likely undergo breaking changes during development. Some of the instructions in the README are aspirational.

## Overview
The Lomb-Scargle periodogram, used for identifying periodicity in irregularly-
spaced observations, is useful but computationally expensive. However, it can be
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
$ pip install niftyls
```

For GPU (CUDA) support:

```console
$ pip install niftyls[cuda]
```


### From source
First, clone the repo:
```console
$ git clone https://www.github.com/flatironinstitute/niftyls
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

<body><!--StartFragment--><pre><div style='color: #000000; background-color: #ffffff; font-family: 'Droid Sans Mono', 'monospace', monospace, monospace; font-size: 14px;'><div><span>(venv) </span><span style='color: #0dbc79; font-weight: bold;'>scclin021:</span><span style='color: #2472c8; font-weight: bold;'>~/nifty-ls</span><span>$ pytest                                                                                                            </span></div><div><span></span><span style='font-weight: bold;'>============================================================= test session starts =============================================================</span></div><div><span></span><span>platform linux -- Python 3.10.13, pytest-8.0.2, pluggy-1.4.0                                                                                   </span></div><div><span>benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=True min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup</span></div><div><span>=False warmup_iterations=100000)                                                                                                               </span></div><div><span>rootdir: /mnt/home/lgarrison/nifty-ls                                                                                                          </span></div><div><span>configfile: pyproject.toml                                                                                                                     </span></div><div><span>plugins: benchmark-4.0.0, asdf-2.15.0, anyio-3.6.2, hypothesis-6.23.1                                                                          </span></div><div><span></span><span style='font-weight: bold;'>collected 10 items                                                                                                                            </span><span> </span></div><div><span>                                                                                                                                               </span></div><div><span>tests/test_ls.py </span><span style='color: #0dbc79;'>..........                                                                                                             [100%]</span><span> </span></div><div><span>                                                                                                                                               </span></div><div><span>                                                                                                                                               </span></div><div><span></span><span style='color: #e5e510;'>----------------------------------------------------------------------------- benchmark '10000': 2 tests --------------------------------------</span></div><div><span>--------------------------------------</span><span>                                                                                                         </span></div><div><span>Name (time in ms)          Min                Max              Mean            StdDev            Median               IQR            Outliers  </span></div><div><span>     OPS            Rounds  Iterations                                                                                                         </span></div><div><span></span><span style='color: #e5e510;'>-----------------------------------------------------------------------------------------------------------------------------------------------</span></div><div><span>--------------------------------------</span><span>                                                                                                         </span></div><div><span>test_nifty[10000]     </span><span style='color: #0dbc79; font-weight: bold;'>  1.9412 (1.0)       4.7566 (1.0)      1.9915 (1.0)      0.1955 (1.0)      1.9629 (1.0)      0.0180 (1.0)    </span><span>     10;24</span><span style='color: #0dbc79; font-weight: bold;'>  </span></div><div><span>502.1457 (1.0)    </span><span>     414           1                                                                                                         </span></div><div><span>test_astropy[10000]   </span><span style='color: #cd3131; font-weight: bold;'>  6.0092 (3.10)     15.4854 (3.26)     6.8011 (3.42)     1.9705 (10.08)    6.0990 (3.11)     0.2403 (13.34)  </span><span>      8;16</span><span style='color: #cd3131; font-weight: bold;'>  </span></div><div><span>147.0349 (0.29)   </span><span>      95           1                                                                                                         </span></div><div><span></span><span style='color: #e5e510;'>-----------------------------------------------------------------------------------------------------------------------------------------------</span></div><div><span>--------------------------------------</span><span>                                                                                                         </span></div><div><span>                                                                                                                                               </span></div><div><span></span><span style='color: #e5e510;'>----------------------------------------------------------------------------- benchmark '100000': 2 tests -------------------------------------</span></div><div><span>-----------------------------------------</span><span>                                                                                                      </span></div><div><span>Name (time in ms)            Min                Max               Mean            StdDev             Median               IQR            Outlie</span></div><div><span>rs      OPS            Rounds  Iterations                                                                                                      </span></div><div><span></span><span style='color: #e5e510;'>-----------------------------------------------------------------------------------------------------------------------------------------------</span></div><div><span>-----------------------------------------</span><span>                                                                                                      </span></div><div><span>test_nifty[100000]     </span><span style='color: #0dbc79; font-weight: bold;'>   9.8104 (1.0)      33.8985 (1.0)      10.5052 (1.0)    </span><span style='color: #cd3131; font-weight: bold;'>  2.6011 (2.64)   </span><span style='color: #0dbc79; font-weight: bold;'>   9.9750 (1.0)      0.0980 (1.0)    </span><span>      3;</span></div><div><span>13</span><span style='color: #0dbc79; font-weight: bold;'>  95.1909 (1.0)    </span><span>      94           1                                                                                                      </span></div><div><span>test_astropy[100000]   </span><span style='color: #cd3131; font-weight: bold;'>  43.7176 (4.46)     47.0181 (1.39)     44.6886 (4.25)   </span><span style='color: #0dbc79; font-weight: bold;'>  0.9865 (1.0)    </span><span style='color: #cd3131; font-weight: bold;'>  44.3489 (4.45)     1.3809 (14.09)  </span><span>       4</span></div><div><span>;0</span><span style='color: #cd3131; font-weight: bold;'>  22.3771 (0.24)   </span><span>      19           1                                                                                                      </span></div><div><span></span><span style='color: #e5e510;'>-----------------------------------------------------------------------------------------------------------------------------------------------</span></div><div><span>-----------------------------------------</span><span>                                                                                                      </span></div><div><span>                                                                                                                                               </span></div><div><span></span><span style='color: #e5e510;'>------------------------------------------------------------------------------------ benchmark '1000000': 2 tests -----------------------------</span></div><div><span>------------------------------------------------------</span><span>                                                                                         </span></div><div><span>Name (time in ms)                Min                   Max                  Mean            StdDev                Median                IQR    </span></div><div><span>        Outliers     OPS            Rounds  Iterations                                                                                         </span></div><div><span></span><span style='color: #e5e510;'>-----------------------------------------------------------------------------------------------------------------------------------------------</span></div><div><span>------------------------------------------------------</span><span>                                                                                         </span></div><div><span>test_nifty[1000000]     </span><span style='color: #0dbc79; font-weight: bold;'>     90.8958 (1.0)        110.8337 (1.0)        101.9764 (1.0)    </span><span style='color: #cd3131; font-weight: bold;'>  7.0199 (2.21)   </span><span style='color: #0dbc79; font-weight: bold;'>    100.1216 (1.0)    </span><span style='color: #cd3131; font-weight: bold;'>  10.0566 (1.</span></div><div><span>68)   </span><span>       3;0</span><span style='color: #0dbc79; font-weight: bold;'>  9.8062 (1.0)    </span><span>       7           1                                                                                         </span></div><div><span>test_astropy[1000000]   </span><span style='color: #cd3131; font-weight: bold;'>  1,092.2903 (12.02)    1,099.2451 (9.92)     1,095.5154 (10.74)  </span><span style='color: #0dbc79; font-weight: bold;'>  3.1811 (1.0)    </span><span style='color: #cd3131; font-weight: bold;'>  1,095.3765 (10.94)  </span><span style='color: #0dbc79; font-weight: bold;'>   5.9920 (1.</span></div><div><span>0)    </span><span>       2;0</span><span style='color: #cd3131; font-weight: bold;'>  0.9128 (0.09)   </span><span>       5           1                                                                                         </span></div><div><span></span><span style='color: #e5e510;'>-----------------------------------------------------------------------------------------------------------------------------------------------</span></div><div><span>------------------------------------------------------</span><span>                                                                                         </span></div><div><span>                                                                                                                                               </span></div><div><span>Legend:                                                                                                                                        </span></div><div><span>  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.                                  </span></div><div><span>  OPS: Operations Per Second, computed as 1 / Mean                                                                                             </span></div><div><span></span><span style='color: #0dbc79;'>============================================================= </span><span style='color: #0dbc79; font-weight: bold;'>10 passed</span><span style='color: #0dbc79;'> in 16.11s =============================================================</span></div></div></pre><!--EndFragment--></body>

## Developer Notes
Early development of the package was done in the https://github.com/dfm/nufft-ls repo.
