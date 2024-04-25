from __future__ import annotations

from importlib import import_module
from typing import Literal, get_args

__all__ = ['available_backends', 'BACKEND_TYPE', 'BACKEND_NAMES']

BACKEND_TYPE = Literal['finufft', 'cufinufft']
BACKEND_NAMES = list(get_args(BACKEND_TYPE))


def available_backends(verbose: bool = False) -> list[str]:
    """Return a list of available backends.  Backends may be unavailable if their dependencies are not installed."""
    backends = []

    for backend in BACKEND_NAMES:
        try:
            # from . import backend
            import_module(f'.{backend}', __package__)
        except ImportError as e:
            if verbose:
                print(f'Backend {backend} is unavailable: {e}')
        else:
            backends.append(backend)

    return backends
