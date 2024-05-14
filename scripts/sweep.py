from pathlib import Path

import asdf
import click
from astropy.table import Table
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import matplotlib.pyplot as plt

from bench import run_one
from bench import METHODS
from bench import get_plot_kwargs

DEFAULT_DTYPE = 'f8'
DEFAULT_LOGN = 3
DEFAULT_METHODS = ['cufinufft', 'finufft', 'astropy']

# TODO: redo using CPU brute as reference instead of astropy brute?
# Want to go to higher N

# Sweep a range of Nf, such that pairs of Nf fall on either side of powers of 2
# This is because Astropy upsamples to FFT frequencies which are powers of 2
DEFAULT_NF_SWEEP = [int(2**i / 5) for i in range(16, 24, 1)]
DEFAULT_NF_SWEEP = sum([[i, i + 1] for i in DEFAULT_NF_SWEEP], [])
print(DEFAULT_NF_SWEEP)

DEFAULT_N_SWEEP = [int(i / 12.5) for i in DEFAULT_NF_SWEEP]
print(DEFAULT_N_SWEEP)


@click.group()
def cli():
    pass


@cli.command('run', context_settings={'show_default': True})
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
    '-o', '--output-file', default='sweep_results.asdf', help='File to save results to'
)
def run(dtype, methods, ref, output_file):
    all_tables = []
    all_methods = list(methods) + [ref]
    for i in range(len(DEFAULT_N_SWEEP)):
        N, Nf = DEFAULT_N_SWEEP[i], DEFAULT_NF_SWEEP[i]
        table = []
        for method in all_methods:
            res = run_one(method, N, Nf, dtype, squeeze=True, time=False)
            print(
                f'{method:13s} N={N:5d}, Nf={Nf:6d}, time={res["firsttime"]:8.4g} sec'
            )
            table.append(res)
        table = Table(table, meta={'N': N, 'Nf': Nf, 'ref': ref})
        all_tables.append(table)

    # save results
    af = asdf.AsdfFile(tree=dict(data=all_tables))
    af.write_to(output_file)

    plot_fname = Path(output_file).with_suffix('.png')
    _analyze(all_tables, fname=plot_fname)


@cli.command()
@click.argument('results_file')
def analyze(results_file):
    with asdf.open(results_file, lazy_load=False, copy_arrays=True) as af:
        tables = af['data']
    plot_fname = Path(results_file).with_suffix('.png')
    _analyze(tables, fname=plot_fname)


def _analyze(all_tables, fname, plot=True):
    # dtype = all_tables[0]['dtype'][0]

    # each table is one N/Nf value
    # for each N and method, establish quartiles
    all_info = []
    for t in range(len(all_tables)):
        table: Table = all_tables[t]
        powers = {row['method']: row['power'] for row in table}
        if t % 2 == 0:
            powers['astropy_worst'] = powers.pop('astropy')
        ref = table.meta['ref']
        for method in powers:
            if method == ref:
                continue
            if (
                'astropy' not in method
                and t > 0
                and table.meta['N'] == all_tables[t - 1].meta['N']
            ):
                # skip duplicate N values
                continue
            # if method == 'cufinufft':
            #     continue
            p1, p2 = powers[method], powers[ref]

            denom = powers[ref]
            nz = denom != 0
            frac = np.abs(p1 - p2) / denom
            frac = np.sort(frac[nz])
            F = len(frac)

            bxp_stats = {
                'med': frac[int(F * 0.5)],
                'q1': frac[int(F * 0.1)],
                'q3': frac[int(F * 0.9)],
                'whislo': frac[int(F * 0.01)],
                'whishi': frac[int(F * 0.99)],
            }

            all_info.append(
                dict(
                    method=method,
                    N=table.meta['N'],
                    Nf=table.meta['Nf'],
                    bxp_stats=bxp_stats,
                )
            )

            print(f'{method} vs {ref}, N={table.meta["N"]}, Nf={table.meta["Nf"]}')
            # pretty-print stats with :.4g
            print(f'\t99% err {bxp_stats["whishi"]*100:.4g}%')
            print(f'\t90% err {bxp_stats["q3"]*100:.4g}%')
            print(f'\tmed err {bxp_stats["med"]*100:.4g}%')

    if plot:
        # _plot_box(all_info, fname)
        _plot_99(all_info, fname)


def _plot_99(all_info, fname):
    """Plot the median and 99th percentile error"""
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.set_yscale('log')
    ax.set_xscale('log')

    all_info = sorted(all_info, key=lambda x: x['bxp_stats']['whishi'], reverse=True)
    methods = [info['method'] for info in all_info]
    Ns = [info['N'] for info in all_info]
    # Nfs = [info['Nf'] for info in all_info]
    # stats = [info['bxp_stats'] for info in all_info]

    lo = [info['bxp_stats']['med'] for info in all_info]
    hi = [info['bxp_stats']['whishi'] for info in all_info]
    o = 0.1
    offsets = {
        'cufinufft': 1.0,
        'finufft': 1.0 - o,
        'astropy': 1.0 + o,
        'astropy_worst': 1.0 + 2 * o,
    }
    lines = {}
    for i in range(len(Ns)):
        plot_kwargs = get_plot_kwargs(methods[i])
        if methods[i] == 'astropy':
            plot_kwargs['label'] = 'astropy'
        x = Ns[i] * offsets[methods[i]]
        line = ax.plot([x, x], (lo[i], hi[i]), **plot_kwargs)
        lines[methods[i]] = line[0]
        ax.plot(x, lo[i], 'o', **plot_kwargs)
        ax.plot(x, hi[i], 'v', **plot_kwargs)

    ax.set_ylim(1e-15)

    # Make dummy entries for the v and o symbols
    cap1 = ax.plot([], [], 'o', color='black', label='Median')
    cap2 = ax.plot([], [], 'v', color='black', label='99th percentile')

    # Add a legend entry that's a line with a solid circle capping one end and a down-triangle (v) capping the other
    ax.legend(
        [
            (cap1[0], cap2[0]),
            lines['astropy'],
            lines['astropy_worst'],
            lines['finufft'],
            lines['cufinufft'],
        ],
        [
            'Median & 99th percentile',
            'astropy',
            'astropy (worst case)',
            'finufft',
            'cufinufft',
        ],
        ncol=1,
        loc='lower right',
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize='small',
    )

    ax.set_xlabel('Number of data points $N$')
    ax.set_ylabel('Fractional error')

    fig.tight_layout()
    fig.savefig(fname)


def _plot_box(all_info, fname):
    """Use matplotlib bxp to plot boxplots of error statistics"""
    fig, ax = plt.subplots()
    ax: plt.Axes
    # ax.set_xscale('log')
    ax.set_yscale('log')

    all_info = sorted(all_info, key=lambda x: x['bxp_stats']['med'], reverse=True)
    # methods = [info['method'] for info in all_info]
    Ns = [info['N'] for info in all_info]
    # Nfs = [info['Nf'] for info in all_info]
    stats = [info['bxp_stats'] for info in all_info]
    positions = np.log10(Ns)

    # boxplot
    ax.bxp(stats, positions=positions, showfliers=False)
    # ax.set_xticks(range(len(all_info)))
    # ax.set_xticklabels([f'{method}\nN={N}, Nf={Nf}' for method, N, Nf in zip(methods, Ns, Nfs)], rotation=45)
    ax.set_ylabel('Fractional error')

    fig.tight_layout()
    fig.savefig(fname)


if __name__ == '__main__':
    cli()
