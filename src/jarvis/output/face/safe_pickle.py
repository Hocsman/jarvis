"""Read a pickle that may only hold plain data and numpy arrays.

Ditto ships its configuration as a pickle, and unpickling runs whatever a
pickle names. This unpickler resolves the few names a numpy array needs and
refuses every other one, so a tampered file fails to load instead of running
code.
"""

import pickle

_ALLOWED = frozenset({
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
})


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def load(path):
    """Unpickle ``path``, raising ``pickle.UnpicklingError`` on anything that
    is not plain data or a numpy array."""
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f).load()
