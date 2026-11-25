# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The PASE family's runner, with the release's two modules faked.

Items routed by length to the two modules, existing outputs kept, checkpoints
passed by path, writes made float32, outputs fitted to the input.
"""

# pylint: disable=redefined-outer-name,unused-argument

import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

import pase


@pytest.fixture
def lengths(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    for name, seconds in (("short.wav", 5), ("long.wav", 25), ("done.wav", 1)):
        sf.write(
            inp / name,
            np.zeros(48000 * seconds, dtype="float32"),
            48000,
            subtype="FLOAT",
        )
    sf.write(
        out / "done.wav",
        np.ones(48000, dtype="float32"),
        48000,
        subtype="FLOAT",
    )
    return inp, out


def test_items_are_routed_by_length_resumed_and_fitted(lengths, monkeypatch):
    inp, out = lengths
    calls = []

    def run_module(repo, module, in_dir, out_dir, device, checkpoints):
        calls.append((module, sorted(os.listdir(in_dir)), device, checkpoints))
        os.makedirs(out_dir)
        # the release: the input's rate, a length of its own
        for name in os.listdir(in_dir):
            sf.write(
                os.path.join(out_dir, name),
                np.full(48000 * 25 - 100, 0.25, "float32"),
                48000,
            )

    monkeypatch.setattr(pase, "run_module", run_module)
    files = sorted(str(p) for p in inp.iterdir())
    assert (pase.render("repo", files[:2], str(out), "cuda:0",
                        {"--ckpt": "c.pt"}) == 0)
    assert [(m, n) for m, n, _, _ in calls] == [("inference.inference_long",
                                                 ["long.wav"])]
    assert calls[0][2:] == ("cuda:0", {"--ckpt": "c.pt"})
    y, rate = sf.read(out / "long.wav", dtype="float32")
    assert rate == 48000 and len(y) == 48000 * 25 and y[-100:].max() == 0
    # already there: not re-run
    assert sf.read(out / "done.wav")[0].min() == 1.0


def test_a_failed_module_fails_its_group_only(lengths, monkeypatch, capsys):
    inp, out = lengths

    def run_module(repo, module, in_dir, out_dir, device, checkpoints):
        if module == "inference.inference_long":
            raise subprocess.CalledProcessError(1, module)
        os.makedirs(out_dir)
        for name in os.listdir(in_dir):
            sf.write(
                os.path.join(out_dir, name),
                np.zeros(48000 * 5, "float32"),
                48000,
            )

    monkeypatch.setattr(pase, "run_module", run_module)
    files = sorted(str(p) for p in inp.iterdir())
    assert pase.render("repo", files, str(out), "cpu", {}) == 1
    printed = capsys.readouterr().out
    assert "FAIL long.wav" in printed and "DONE 2/3" in printed
    assert (out / "short.wav").exists() and not (out / "long.wav").exists()


def test_a_missing_checkpoint_stops_before_the_release_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="setupfield"):
        pase.run_module(
            str(tmp_path),
            "inference.inference",
            str(tmp_path),
            str(tmp_path),
            "cpu",
            {"--ckpt": str(tmp_path / "missing.pt")},
        )


def test_the_runner_makes_the_releases_default_writes_float32(tmp_path):
    (tmp_path / "fakerelease.py").write_text(
        "import sys, numpy as np, soundfile as sf\n"
        "out = sys.argv[sys.argv.index('-O') + 1]\n"
        "x = np.full(100, 0.123456789, dtype='float32')\n"
        "sf.write(out + '/default.wav', x, 16000)\n"
        "sf.write(out + '/explicit.wav', x, 16000, subtype='PCM_24')\n"
        "sf.write(out + '/positional.wav', x, 16000, 'PCM_16')\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    subprocess.run(
        [sys.executable, "-c", pase.RUNNER, "fakerelease", "-O",
         str(tmp_path)],
        env=env,
        check=True,
    )
    assert sf.info(tmp_path / "default.wav").subtype == "FLOAT"
    x, _ = sf.read(tmp_path / "default.wav", dtype="float32")
    assert x[0] == np.float32(0.123456789)
    assert sf.info(tmp_path / "explicit.wav").subtype == "PCM_24"
    assert sf.info(tmp_path / "positional.wav").subtype == "PCM_16"
