from pathlib import Path

import click
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

from bench import run_one
from bench import METHODS
from bench import get_plot_kwargs

DEFAULT_DTYPE = 'f8'
DEFAULT_LOGN = 3
DEFAULT_METHODS = [
    'cufinufft',
    'cufinufft_chi2',
    'finufft',
    'astropy',
    'finufft_chi2',
    'astropy_fastchi2',
]


@click.group()
def cli():
    pass


@cli.command('run', context_settings={'show_default': True})
@click.option('-logN', 'logN', type=float, default=DEFAULT_LOGN, help='log10 of N data')
@click.option('-Nf', 'Nf', type=int, default=None, help='Number of modes')
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
    '--ref',
    '-r',
    default='astropy_brute',
    help='reference method',
    type=click.Choice(METHODS),
)
@click.option(
    '-o',
    '--output-file',
    default='compare_results.asdf',
    help='File to save results to',
)
def run(logN, Nf, dtype, methods, ref, output_file):
    N = int(10**logN)

    all_res = []
    all_methods = list(methods) + [ref]
    for method in all_methods:
        res = run_one(method, N, Nf, dtype, squeeze=True, time=False)
        all_res.append(res)

    all_res = Table(all_res, meta={'N': N, 'ref': ref})

    # save results
    all_res.write(output_file)

    plot_fname = Path(output_file).with_suffix('.png')
    _analyze(all_res, fname=plot_fname)


@cli.command()
@click.argument('results_file')
def analyze(results_file):
    results = Table.read(results_file)
    plot_fname = Path(results_file).with_suffix('.png')
    _analyze(results, fname=plot_fname)


def _analyze(all_res, fname, plot=True):
    dtype = all_res['dtype'][0]
    f0 = all_res['f0'][0]
    df = all_res['df'][0]
    N = all_res.meta['N']
    ref = all_res.meta['ref']
    Nf = len(all_res[0]['power'])
    all_res = {
        row['method']: row['power']
        for row in sorted(all_res, key=lambda x: x['power'][-1], reverse=True)
    }

    for k in all_res:
        if k == ref:
            continue
        p1, p2 = all_res[k], all_res[ref]

        rms_err = np.nanmean((p2 - p1) ** 2, dtype=np.float64) ** 0.5
        rms_mean = np.nanmean(((p1 + p2) / 2) ** 2, dtype=np.float64) ** 0.5

        print(f'{k} vs {ref}')
        denom = all_res[ref]
        nz = denom != 0
        frac = np.abs(p1 - p2) / denom
        frac = frac[nz]
        print(f'\tmax err {np.nanmax(frac) * 100:.4g}%')
        print(f'\t99% err {np.sort(frac)[int(len(frac) * 0.99)] * 100:.4g}%')
        print(f'\t90% err {np.sort(frac)[int(len(frac) * 0.9)] * 100:.4g}%')
        print(f'\t75% err {np.sort(frac)[int(len(frac) * 0.75)] * 100:.4g}%')
        print(f'\t50% err {np.sort(frac)[int(len(frac) * 0.50)] * 100:.4g}%')
        print(f'\trms err / mean {rms_err / rms_mean * 100:.4g}%')

    if plot:
        freq = f0 + df * np.arange(Nf)
        fig, ax = plt.subplots()
        denom = all_res[ref]
        for k in all_res:
            if k == ref:
                continue
            diff = all_res[k] - all_res[ref]
            plot_kwargs = get_plot_kwargs(k)
            ax.plot(freq, np.abs(diff) / denom, **plot_kwargs)
            # ax.plot(freq, all_res[k], **plot_kwargs)
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('frequency')
        # ax.set_ylabel(r'$|\mathrm{LS} - \mathrm{LS}_{\rm ' + ref + r'}| / \mathrm{LS}_{\rm ' + ref + r'}$')
        ax.set_ylabel('Frac. error')
        ax.tick_params(right=True, top=True, which='both')
        if dtype == 'float32':
            ax.set_ylim(bottom=1e-7)

        lines = [
            rf'$N = 10^{{{np.log10(N):.2g}}}$',
            '$N_f \\approx 12N$',
        ]
        lines = '\n'.join(lines)
        ax.annotate(
            lines,
            xy=(0.99, 0.01),
            xycoords='axes fraction',
            ha='right',
            va='bottom',
        )

        fig.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    cli()
