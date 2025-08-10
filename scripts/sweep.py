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
DEFAULT_METHODS = [
    'cufinufft',
    'cufinufft_chi2',
    'finufft',
    'astropy',
    'finufft_chi2',
    'astropy_fastchi2',
]

# TODO: redo using CPU brute as reference instead of astropy brute?
# Want to go to higher N


# Function to generate sweep ranges based on min and max parameters
def generate_sweep_ranges(min_nf, max_nf):
    nf_sweep = [int(2**i / 5) for i in range(min_nf, max_nf, 1)]
    nf_sweep = sum([[i, i + 1] for i in nf_sweep], [])
    n_sweep = [int(i / 12.5) for i in nf_sweep]
    return n_sweep, nf_sweep


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
@click.option(
    '--min-nf', default=16, help='Minimum power for generating NF sweep range', type=int
)
@click.option(
    '--max-nf', default=24, help='Maximum power for generating NF sweep range', type=int
)
@click.option(
    '--max-cunf',
    default=None,
    help='Maximum power for CUDA methods (defaults to max-nf)',
    type=int,
)
def run(dtype, methods, ref, output_file, min_nf, max_nf, max_cunf):
    if max_cunf is None:
        max_cunf = max_nf
    # Check if any CUDA methods are present in methods or reference
    all_methods = list(methods) + [ref]
    using_cuda = any('cu' in method for method in all_methods)

    # Normal sweep range
    n_sweep, nf_sweep = generate_sweep_ranges(min_nf, max_nf)
    print(f'Regular N sweep: {n_sweep}')
    print(f'Regular NF sweep: {nf_sweep}')

    # CUDA-specific sweep range
    cuda_n_sweep, cuda_nf_sweep = None, None
    if using_cuda:
        cuda_n_sweep, cuda_nf_sweep = generate_sweep_ranges(
            min_nf, min(max_nf, max_cunf)
        )
        print(f'CUDA N sweep: {cuda_n_sweep}')
        print(f'CUDA NF sweep: {cuda_nf_sweep}')
    all_tables = []
    for i in range(len(n_sweep)):
        N, Nf = n_sweep[i], nf_sweep[i]
        table = []
        for method in all_methods:
            # Skip CUDA methods
            if 'cu' in method and using_cuda:
                continue
            res = run_one(method, N, Nf, dtype, squeeze=True, time=False)
            print(
                f'{method:13s} N={N:5d}, Nf={Nf:6d}, time={res["firsttime"]:8.4g} sec'
            )
            table.append(res)

        if table:  # Only add the table if it has any entries
            table = Table(table, meta={'N': N, 'Nf': Nf, 'ref': ref})
            all_tables.append(table)

    # Process CUDA-specific N/Nf pairs
    if using_cuda and cuda_n_sweep is not None:
        for i in range(len(cuda_n_sweep)):
            N, Nf = cuda_n_sweep[i], cuda_nf_sweep[i]
            table = []
            for method in all_methods:
                if 'cu' not in method:
                    continue
                res = run_one(method, N, Nf, dtype, squeeze=True, time=False)
                print(
                    f'{method:13s} N={N:5d}, Nf={Nf:6d}, time={res["firsttime"]:8.4g} sec'
                )
                table.append(res)

            if table:
                existing_table = None
                for existing in all_tables:
                    if existing.meta['N'] == N and existing.meta['Nf'] == Nf:
                        existing_table = existing
                        break

                if existing_table is not None:
                    for row in table:
                        existing_table.add_row(row)
                else:
                    table = Table(table, meta={'N': N, 'Nf': Nf, 'ref': ref})
                    all_tables.append(table)

    # save results
    af = asdf.AsdfFile(tree=dict(data=all_tables))
    af.write_to(output_file)

    plot_fname = Path(output_file).with_suffix('.png')
    _analyze(all_tables, fname=plot_fname)


@cli.command()
@click.argument('results_file')
@click.option('--paper', is_flag=True, help='Generate plots for paper')
def analyze(results_file, paper=False):
    with asdf.open(results_file, lazy_load=False) as af:
        tables = af['data']
    plot_fname = Path(results_file).with_suffix('.png')
    _analyze(tables, fname=plot_fname, paper=paper)


def _analyze(all_tables, fname, plot=True, paper=False):
    # dtype = all_tables[0]['dtype'][0]

    # each table is one N/Nf value
    # for each N and method, establish quartiles
    all_info = []
    for t in range(len(all_tables)):
        table: Table = all_tables[t]
        powers = {row['method']: row['power'] for row in table}
        if t % 2 == 0 and 'astropy' in powers:
            powers['astropy_worst'] = powers.pop('astropy')
        elif t % 2 == 0 and 'astropy_fastchi2' in powers:
            powers['astropy_fastchi2_worst'] = powers.pop('astropy_fastchi2')
        ref = table.meta['ref']
        for method in powers:
            if method == ref:
                continue
            if (
                'astropy' not in method
                and t > 0
                and table.meta['N'] == all_tables[t - 1].meta['N']
            ) or (
                'astropy_fastchi2' not in method
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
            print(f'\t99% err {bxp_stats["whishi"] * 100:.4g}%')
            print(f'\t90% err {bxp_stats["q3"] * 100:.4g}%')
            print(f'\tmed err {bxp_stats["med"] * 100:.4g}%')

    if plot:
        # _plot_box(all_info, fname)
        _plot_99(all_info, fname, paper=paper)


def _plot_99(all_info, fname, paper=False):
    """Plot the median and 99th percentile error"""
    figsize = (w := 3.8, w / 1.3) if paper else (6, 4)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
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
    o = 0.125
    offsets = {
        'cufinufft': 1.0,
        'finufft': 1.0 - o,
        'astropy': 1.0 + o,
        'astropy_worst': 1.0 + 2 * o,
        'cufinufft_chi2': 1.0,
        'finufft_chi2': 1.0 - o,
        'astropy_fastchi2': 1.0 + o,
        'astropy_fastchi2_worst': 1.0 + 2 * o,
    }
    lines = {}
    for i in range(len(Ns)):
        plot_kwargs = get_plot_kwargs(methods[i])
        if methods[i] == 'astropy':
            plot_kwargs['label'] = 'astropy'
        elif methods[i] == 'astropy_fastchi2':
            plot_kwargs['label'] = 'astropy_fastchi2'
        x = Ns[i] * offsets[methods[i]]
        line = ax.plot([x, x], (lo[i], hi[i]), **plot_kwargs)
        lines[methods[i]] = line[0]
        ax.plot(x, lo[i], 'o', **plot_kwargs)
        ax.plot(x, hi[i], 'v', **plot_kwargs)

    ax.set_ylim(1e-17)

    # Make dummy entries for the v and o symbols
    cap1 = ax.plot([], [], 'o', color='black', label='Median')
    cap2 = ax.plot([], [], 'v', color='black', label='99th percentile')

    legend_elements = [(cap1[0], cap2[0])]
    legend_labels = ['50th & 99th percentile']

    method_priority = [
        'astropy',
        'astropy_worst',
        'astropy_fastchi2',
        'astropy_fastchi2_worst',
        'finufft',
        'finufft_chi2',
        'cufinufft',
        'cufinufft_chi2',
    ]

    for method in method_priority:
        if method in lines:
            legend_elements.append(lines[method])
            method_kwargs = get_plot_kwargs(method)
            legend_labels.append(method_kwargs.get('label', method))

    ax.legend(
        legend_elements,
        legend_labels,
        ncol=2 if paper else 1,
        loc='lower right',
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize='small',
    )

    ax.set_xlabel('Number of data points ($N_d$)')
    ax.set_ylabel('Fractional error')

    # Add grid for better readability
    ax.grid(which='both', linestyle='--', alpha=0.5)

    # Improve tick formatting
    ax.tick_params(right=True, top=True, which='both')

    # Format y-axis ticks as powers of 10
    logymaxtick = int(np.log10(ax.get_ylim()[1]))
    logymintick = int(np.ceil(np.log10(ax.get_ylim()[0])))
    numyticks = logymaxtick - logymintick
    ticks = np.logspace(logymintick, logymaxtick, numyticks + 1)
    ax.set_yticks(
        ticks,
        labels=[
            f'$10^{{{np.log10(t):.0f}}}$' if i % 3 == 0 else ''
            for i, t in enumerate(ticks)
        ],
    )

    fig.savefig(fname)
    fig.savefig(Path(fname).with_suffix('.pdf'))


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

    fig.savefig(fname)


if __name__ == '__main__':
    cli()
