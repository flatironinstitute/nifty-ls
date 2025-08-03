from __future__ import annotations

from importlib import import_module
from typing import Literal, get_args, Optional, List

__all__ = [
    'available_backends',
    'BACKEND_TYPE',
    'BACKEND_NAMES',
    'CHI2_BACKEND_NAMES',
    'STANDARD_BACKEND_NAMES',
    'HETEROBATCH_BACKEND_NAMES',
]

CHI2_BACKEND_NAMES = ['finufft_chi2', 'cufinufft_chi2']
STANDARD_BACKEND_NAMES = ['finufft', 'cufinufft']
BACKEND_TYPE = Literal['auto', 'finufft', 'finufft_chi2', 'cufinufft', 'cufinufft_chi2']
BACKEND_NAMES = list(get_args(BACKEND_TYPE))
HETEROBATCH_BACKEND_TYPE = Literal['auto', 'finufft_heterobatch']
HETEROBATCH_BACKEND_NAMES = list(get_args(HETEROBATCH_BACKEND_TYPE))


def available_backends(verbose: bool = False, backend_names: Optional[List[str]] = None) -> list[str]:
    """Return a list of available backends.  Backends may be unavailable if their dependencies are not installed."""
    backends = []
    # Default BACKEND_NAMES
    if backend_names is None:
        backend_names = BACKEND_NAMES

    for backend in backend_names:
        if backend == 'auto':
            # 'auto' is a special case, it is not a backend but a mode to select the best available backend
            backends.append(backend)
            continue
        try:
            # from . import backend
            import_module(f'.{backend}', __package__)
        except ImportError as e:
            if verbose:
                print(f'[nifty-ls] Backend {backend} is unavailable: {e}')
        else:
            backends.append(backend)

    return backends
