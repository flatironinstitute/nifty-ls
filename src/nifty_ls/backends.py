from __future__ import annotations

from importlib import import_module


BACKEND_NAMES = ['finufft', 'cufinufft']


def available_backends(verbose: bool = True) -> list[str]:
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
