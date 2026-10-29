"""The talking face's files: the portrait, the model weights and the engines' key
(`src/jarvis/output/face/face.spec.md`, "The model", "Model files and engines")."""

import hashlib

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

    def test_a_missing_file_is_refused(self, tmp_path):
        pytest.importorskip("cv2")
        with pytest.raises(portrait.UnreadablePortrait):
            portrait.load(str(tmp_path / "nowhere.png"))

    def test_a_file_that_is_not_an_image_is_refused(self, tmp_path):
        pytest.importorskip("cv2")
        bad = tmp_path / "portrait.png"
        bad.write_bytes(b"not an image")

        with pytest.raises(portrait.UnreadablePortrait):
            portrait.load(str(bad))


def _fake_files(monkeypatch, contents):
    files = tuple((name, hashlib.sha256(data).hexdigest(), len(data)) for name, data in contents.items())
    monkeypatch.setattr(weights, "FILES", files)


class TestWeights:
    def test_only_missing_files_are_fetched(self, tmp_path, monkeypatch):
        _fake_files(monkeypatch, {"a/one.bin": b"one", "two.bin": b"two!"})
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.bin").write_bytes(b"one")
        asked = []

        def download(repo_id, filename, revision, local_dir):
            asked.append((repo_id, filename, revision))
            path = tmp_path / filename
            path.write_bytes(b"two!")
            return str(path)

        weights.fetch(tmp_path, download=download)

        assert asked == [(weights.REPO, "two.bin", weights.REVISION)]
        assert weights.missing(tmp_path) == []

    def test_a_file_that_does_not_match_its_digest_is_deleted(self, tmp_path, monkeypatch):
        _fake_files(monkeypatch, {"model.bin": b"the real one"})

        def download(repo_id, filename, revision, local_dir):
            path = tmp_path / filename
            path.write_bytes(b"something else")
            return str(path)

        with pytest.raises(RuntimeError, match="digest"):
            weights.fetch(tmp_path, download=download)
        assert not (tmp_path / "model.bin").exists()

    def test_a_failed_download_names_the_file(self, tmp_path, monkeypatch):
        _fake_files(monkeypatch, {"model.bin": b"x"})

        def download(**kwargs):
            raise OSError("network unreachable")

        with pytest.raises(RuntimeError, match="model.bin"):
            weights.fetch(tmp_path, download=download)

    def test_the_real_list_is_pinned(self):
        assert len(weights.REVISION) == 40
        assert all(len(sha) == 64 and size > 0 for _name, sha, size in weights.FILES)
        assert weights.CONFIG in {name for name, _sha, _size in weights.FILES}


class TestEngineKey:
    BASE = ("GPU A", (12, 0), "11.4.0", 100, 200)

    def test_the_same_machine_finds_the_same_engines(self):
        assert engines.engine_key(*self.BASE) == engines.engine_key(*self.BASE)

    @pytest.mark.parametrize("index, other", [(0, "GPU B"), (1, (8, 9)), (2, "10.0.0"), (3, 101), (4, 201)])
    def test_another_gpu_tensorrt_or_weights_builds_new_engines(self, index, other):
        changed = list(self.BASE)
        changed[index] = other

        assert engines.engine_key(*changed) != engines.engine_key(*self.BASE)
