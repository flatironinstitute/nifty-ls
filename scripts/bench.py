from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import nifty_ls.finufft

from astropy.io import ascii
from astropy.table import Table
import click

import matplotlib.pyplot as plt
import numpy as np
import timeit

import nifty_ls
import astropy.timeseries.periodograms.lombscargle.implementations.fast_impl as astropy_impl

# The Gowanlock+ paper uses N_t=3554 as their single-object dataset.
DEFAULT_N = 3554
DEFAULT_NF = None  # 10**5
DEFAULT_DTYPE = 'f8'
DEFAULT_METHODS = ['cufinufft', 'finufft', 'astropy', 'finufft_par']
NTHREAD_MAX = nifty_ls.utils.get_avail_cpus()
DEFAULT_FFTW = nifty_ls.finufft.FFTW_MEASURE
DEFAULT_EPS = 1e-9


def do_nifty_finufft(*args, **kwargs):
    return nifty_ls.finufft.lombscargle(
        *args, **kwargs, finufft_kwargs={'fftw': DEFAULT_FFTW, 'eps': DEFAULT_EPS}
    )


def do_nifty_cufinufft(*args, **kwargs):
    return nifty_ls.cufinufft.lombscargle(
        *args, **kwargs, cufinufft_kwargs={'eps': DEFAULT_EPS}
    )


def do_astropy(t, y, dy, fmin, df, Nf, **astropy_kwargs):
    f0 = fmin
    y = np.atleast_2d(y)
    dy = np.atleast_2d(dy)
    for i in range(y.shape[0]):
        power = astropy_impl.lombscargle_fast(
            t, y[i], dy=dy[i], f0=f0, df=df, Nf=Nf, **astropy_kwargs
        )
    return power  # just last power for now


def do_winding(t, y, dy, fmin, df, Nf, center_data=True, fit_mean=True, **kwargs):
    power = np.empty(Nf, dtype=np.float64)
    nifty_ls.cpu_helpers.compute_winding(
        power, t, y, dy, fmin, df, center_data, fit_mean, **kwargs
    )
    return power


METHODS = {
    'finufft_par': do_nifty_finufft,
    'finufft': lambda *args, **kwargs: do_nifty_finufft(*args, **kwargs, nthreads=1),
    'cufinufft': do_nifty_cufinufft,
    'astropy': do_astropy,
    'astropy_brute': lambda *args, **kwargs: do_astropy(*args, **kwargs, use_fft=False),
    'winding': do_winding,
}


def run_one(
    method,
    N,
    Nf,
    dtype,
    batch_size=1,
    seed=5043,
    squeeze=False,
    fmax=None,
    time=True,
    **kwargs,
):
    dtype = np.dtype(dtype).type
    rng = np.random.default_rng(seed)

    # Generate fake data
    t = np.sort(rng.uniform(0, 2 * np.pi, N).astype(dtype)) * 1000
    y = np.sin(t * rng.uniform(0.1, 10, batch_size).reshape(-1, 1)).astype(dtype)
    dy = rng.normal(size=(batch_size, N)).astype(dtype)

    if squeeze and batch_size == 1:
        y = y[0]
        dy = dy[0]

    f0, df, Nf = nifty_ls.utils.validate_frequency_grid(None, fmax, Nf, t)

    def func():
        return METHODS[method](t, y, dy, f0, df, Nf, **kwargs)

    res = {
        'method': method,
        'Nf': Nf,
        'dtype': dtype.__name__,
        'f0': f0,
        'df': df,
        'N': N,
    }

    # warmup, and get result
    t1 = -timeit.default_timer()
    res['power'] = func()
    t1 += timeit.default_timer()
    res['firsttime'] = t1

    if time:
        nrep, tot_time = timeit.Timer(func).autorange()
        t = tot_time / nrep
        res['time'] = t

    return res


def get_plot_kwargs(method, nthread_max=NTHREAD_MAX):
    if method == 'finufft_par':
        label = (
            'nifty-ls (finufft)'
            if nthread_max == 1
            else 'nifty-ls (finufft, multi-threaded)'
        )
        color = 'C1'
        ls = '--'
    elif method == 'finufft':
        label = 'nifty-ls (finufft)'
        color = 'C1'
        ls = '-'
    elif method == 'cufinufft':
        label = 'nifty-ls (cufinufft)'
        color = 'C2'
        ls = '-'
    elif method == 'astropy':
        label = r'astropy (${\tt fast}$ method)'
        color = 'C0'
        ls = '-'
    elif method == 'astropy_worst':
        label = r'astropy (worst case)'
        color = 'C0'
        ls = '--'
    else:
        label = method
        color = 'C3'
        ls = '-'

    return {'label': label, 'color': color, 'ls': ls}


@click.group()
def cli():
    pass


@cli.command('bench', context_settings={'show_default': True})
@click.option('-logmin', type=float, default=4, help='log10 of min number of modes')
@click.option('-logmax', type=float, default=7, help='log10 of max number of modes')
@click.option('-logdelta', type=float, default=1, help='Spacing in log10 for Nf values')
@click.option(
    '-dtype', default=DEFAULT_DTYPE, help='dtype', type=click.Choice(('f4', 'f8'))
)
@click.option(
    '--method',
    '-m',
    'methods',
    default=DEFAULT_METHODS,
    help='methods to run',
    multiple=True,
    type=click.Choice(METHODS),
)
@click.option(
    '--sweep',
    '-s',
    default='Nf',
    help='Which parameter to sweep',
    type=click.Choice(('Nf', 'N')),
)
@click.option(
    '--batch-size', '-b', default=1, help='Batch size (number of periodograms)'
)
@click.option(
    '-o', '--results-file', default='bench_results.ecsv', help='File to save results to'
)
def bench(
    logmin,
    logmax,
    logdelta,
    dtype,
    methods=DEFAULT_METHODS,
    sweep='Nf',
    results_file='bench_results.ecsv',
    batch_size=1,
):
    # process args
    dtype = np.dtype(dtype).type

    print(
        f'Running with {sweep=}, {logmin=}, {logmax=}, {logdelta=}, dtype {dtype.__name__}, {batch_size=}'
    )

    all_N = np.logspace(
        logmin, logmax, int((logmax - logmin) / logdelta) + 1, dtype=int
    )

    all_res = []
    for method in methods:
        for V in all_N:
            if sweep == 'Nf':
                Nf = V
                N = DEFAULT_N
            elif sweep == 'N':
                N = V
                Nf = DEFAULT_NF

            res = run_one(method, N, Nf, dtype, batch_size=batch_size)

            print(f'{method} took {res["time"]:.4g} sec ({Nf=})')
            all_res.append(res)

    all_res = Table(
        all_res,
        meta={
            'sweep': sweep,
            'nthread_max': NTHREAD_MAX,
            'auto_Nf': DEFAULT_NF is None and sweep == 'N',
            'batch_size': batch_size,
        },
    )

    # compare(all_res)

    del all_res['power']
    all_res.write(results_file, overwrite=True)

    plot_fname = Path(results_file).with_suffix('.png')
    _plot(all_res, fname=plot_fname)


@cli.command()
@click.argument('results_file')
def plot(results_file):
    results = ascii.read(results_file)
    plot_fname = Path(results_file).with_suffix('.png')
    _plot(results, fname=plot_fname)


def _plot(all_res: Table, sort=True, fname='bench_results.png'):
    all_res = all_res.group_by('method')
    fig, ax = plt.subplots()
    ax: plt.Axes
    sweep = all_res.meta['sweep']
    const = 'N' if sweep == 'Nf' else 'Nf'
    var = {'N': 'Nf', 'Nf': 'N'}[const]
    const_desc = {'N': 'Number of data points', 'Nf': 'Number of frequencies'}[const]

    dtype = all_res['dtype'][0]  # noqa: F841
    N_const = all_res['N'][0]  # noqa: F841
    auto_Nf = all_res.meta['auto_Nf']
    Nf_const = all_res['Nf'][0]  # noqa: F841

    if sort:
        groups = sorted(all_res.groups, key=lambda g: g['time'].max(), reverse=True)
    else:
        groups = all_res.groups
    for group in groups:
        method = group['method'][0]
        plot_kwargs = get_plot_kwargs(method, all_res.meta['nthread_max'])
        ax.plot(group[var], group['time'], marker='o', **plot_kwargs)
        lines = []
        if all_res.meta['batch_size'] > 1:
            lines.append(f'Batch size = {all_res.meta["batch_size"]}')

        if auto_Nf and var == 'N':
            lines.append(f'{const_desc} $\\approx 12{var}$')
        else:
            lines.append(
                f'{const_desc}: {eval(const+"_const")}',
            )
        lines = '\n'.join(lines)
        ax.annotate(
            lines,
            xy=(0.99, 0.01),
            xycoords='axes fraction',
            ha='right',
            va='bottom',
        )

    xline = all_res[var].max()
    yline = all_res['time'][all_res[var] == xline].min()

    ax.axline(
        (xline, yline / 2), slope=1, label='Linear scaling', linestyle='--', color='k'
    )

    ax.tick_params(right=True, top=True, which='both')

    ax.legend()
    if sweep == 'Nf':
        ax.set_xlabel('Number of frequencies')
    elif sweep == 'N':
        ax.set_xlabel('Number of data points $N$')
    ax.set_ylabel('Time (seconds)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    cli()
