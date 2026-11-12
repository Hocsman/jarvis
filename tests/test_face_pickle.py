"""Ditto's configuration pickle is read with numpy arrays allowed and nothing else
(`src/jarvis/output/face/face.spec.md`, "The model")."""

import pickle

import numpy as np
import pytest

from jarvis.output.face import safe_pickle


def test_plain_data_and_numpy_arrays_load(tmp_path):
    cfg = {"base_cfg": {"x": [1, 2.5, "a", None, True]}, "mean": np.arange(6, dtype=np.float32).reshape(2, 3)}
    path = tmp_path / "cfg.pkl"
    path.write_bytes(pickle.dumps(cfg))

    got = safe_pickle.load(path)

    assert got["base_cfg"] == cfg["base_cfg"]
    np.testing.assert_array_equal(got["mean"], cfg["mean"])


class _RunsCode:
    def __reduce__(self):
        import os
        return (os.getenv, ("PATH",))


def test_a_pickle_that_names_anything_else_is_refused_without_running_it(tmp_path):
    path = tmp_path / "cfg.pkl"
    path.write_bytes(pickle.dumps({"payload": _RunsCode()}))

    with pytest.raises(pickle.UnpicklingError, match="refused os"):
        safe_pickle.load(path)
