"""The talking face's files: the portrait, the model weights and the engines' key
(`src/jarvis/output/face/face.spec.md`, "The model", "Model files and engines")."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from jarvis.output.face import engines, portrait, weights


class TestPortrait:
    def test_the_picture_is_widened_around_its_own_pixels(self):
        cv2 = pytest.importorskip("cv2")
        img = np.random.default_rng(1).integers(0, 255, (300, 200, 3), np.uint8)

        wide = portrait.widen(img)
        pad = round(portrait.WIDEN * 300)
        ramp = int(portrait.RAMP * 300) + 2

        assert wide.shape == (300 + 2 * pad, 200 + 2 * pad, 3)
        middle = wide[pad + ramp:pad + 300 - ramp, pad + ramp:pad + 200 - ramp]
        np.testing.assert_array_equal(middle, img[ramp:300 - ramp, ramp:200 - ramp])

    def test_a_missing_file_is_refused_without_its_path_in_the_reason(self, tmp_path):
        with pytest.raises(portrait.UnreadablePortrait) as refused:
            portrait.load(str(tmp_path / "nowhere.png"))
        assert "nowhere" not in str(refused.value)

    def test_a_file_that_is_not_an_image_is_refused(self, tmp_path):
        pytest.importorskip("cv2")
        bad = tmp_path / "portrait.png"
        bad.write_bytes(b"not an image")

        with pytest.raises(portrait.UnreadablePortrait):
            portrait.load(str(bad))


def _fake_files(monkeypatch, contents):
    files = tuple((name, hashlib.sha256(data).hexdigest(), len(data)) for name, data in contents.items())
    monkeypatch.setattr(weights, "FILES", files)


def _fake_download(tmp_path, data, asked=None):
    def download(repo_id, filename, revision, local_dir, token):
        if asked is not None:
            asked.append((repo_id, filename, revision, token))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data[filename])
        return str(path)
    return download


class TestWeights:
    def test_only_missing_files_are_fetched_without_a_token_and_moved_into_place(self, tmp_path, monkeypatch):
        contents = {"a/one.bin": b"one", "two.bin": b"two!"}
        _fake_files(monkeypatch, contents)
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.bin").write_bytes(b"one")
        asked = []

        weights.fetch(tmp_path, download=_fake_download(tmp_path, contents, asked))

        assert asked == [(weights.REPO, "two.bin", weights.REVISION, False)]
        assert (tmp_path / "two.bin").read_bytes() == b"two!"
        assert weights.missing(tmp_path) == []

    def test_a_file_that_does_not_match_its_digest_never_reaches_its_place(self, tmp_path, monkeypatch):
        _fake_files(monkeypatch, {"model.bin": b"the real one"})

        with pytest.raises(RuntimeError, match="digest"):
            weights.fetch(tmp_path, download=_fake_download(tmp_path, {"model.bin": b"something else"}))
        assert not (tmp_path / "model.bin").exists()
        assert not any(p.is_file() for p in tmp_path.rglob("model.bin"))

    def test_a_failed_download_names_the_file(self, tmp_path, monkeypatch):
        _fake_files(monkeypatch, {"model.bin": b"x"})

        def download(**kwargs):
            raise OSError("network unreachable")

        with pytest.raises(RuntimeError, match="model.bin"):
            weights.fetch(tmp_path, download=download)

    def test_the_real_list_is_pinned(self):
        names = {name for name, _sha, _size in weights.FILES}
        assert len(weights.REVISION) == 40
        assert all(len(sha) == 64 and size > 0 for _name, sha, size in weights.FILES)
        assert {weights.CONFIG, weights.WARP, weights.DECODER} <= names


class TestEngineKey:
    BASE = ("GPU A", (12, 0), "11.4.0", "a" * 64, "b" * 64)

    def test_the_same_machine_finds_the_same_engines(self):
        assert engines.engine_key(*self.BASE) == engines.engine_key(*self.BASE)

    @pytest.mark.parametrize("index, other", [(0, "GPU B"), (1, (8, 9)), (2, "10.0.0"), (3, "c" * 64), (4, "d" * 64)])
    def test_another_gpu_tensorrt_or_weights_builds_new_engines(self, index, other):
        changed = list(self.BASE)
        changed[index] = other

        assert engines.engine_key(*changed) != engines.engine_key(*self.BASE)


class TestEngineCache:
    @pytest.fixture
    def gpu(self, monkeypatch):
        import types

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            get_device_name=lambda i: "GPU A", get_device_capability=lambda i: (12, 0)))
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="11.4.0"))

    def test_engines_are_built_once_then_found(self, tmp_path, gpu):
        built = []

        def build(warp_pth, decoder_pth, warp_engine, decoder_engine):
            built.append((warp_pth, decoder_pth))
            warp_engine.write_bytes(b"warp")
            decoder_engine.write_bytes(b"decoder")

        first = engines.ensure_engines(tmp_path, build=build)
        second = engines.ensure_engines(tmp_path, build=build)

        assert first == second
        assert built == [(tmp_path / weights.WARP, tmp_path / weights.DECODER)]
        assert all(path.read_bytes() for path in first)
